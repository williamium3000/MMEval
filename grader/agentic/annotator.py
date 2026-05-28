"""Agentic span-level hallucination annotator.

Three-step flow per (q, a) round:
  A. Decompose the answer into atomic visual claims  (text-only LLM, cheap).
  B. Look each claim up in the scene graph             (deterministic Python).
  C. Vision-judge call (gpt-5) with the ORIGINAL image + a copy ANNOTATED with
     bboxes of sg-matched relevant objects, plus a text block listing those
     bboxes + attributes + per-claim sg verdicts. The judge produces the final
     hallucination spans (with per-span confidence + reason), free to override
     sg when it sees otherwise. Cached candidates can be re-thresholded
     without re-calling the API.

Output schema matches the human files in work_dirs/human/result/dyna-v18.
"""

import os
import re
import json
import base64
import hashlib
import threading
import difflib
import time

from dotenv import load_dotenv
from openai import OpenAI

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from grader.agentic.claims import decompose                  # noqa: E402
from grader.agentic.sg_lookup import evaluate_claims         # noqa: E402
from grader.agentic.sg_join import get_sg                    # noqa: E402
from grader.agentic.image_annotate import annotate           # noqa: E402

# Bump on any prompt / decoding change to invalidate the on-disk cache. The
# string is opaque; the cache_v2 -> cache rename preserves all existing entries.
PROMPT_VERSION = "v2agentic4"

MODEL = os.environ.get("AGENTIC_MODEL", "gpt-5")
REASONING_EFFORT = os.environ.get("AGENTIC_REASONING_EFFORT", "low")
CACHE_DIR = os.environ.get(
    "AGENTIC_CACHE_DIR", os.path.join(os.path.dirname(__file__), ".cache")
)
# Confidence filter applied at READ time. "high" = only certain hallucinations;
# "medium" = also keep likely-but-not-certain; "low" = keep all candidates.
# Default "high" to suppress false positives like "X is being used to play the
# game" / "there's a passage for boats" where the claim is plausibly true.
_CONF_ORDER = {"high": 3, "medium": 2, "low": 1}
CONFIDENCE_THRESHOLD = os.environ.get("AGENTIC_CONFIDENCE_THRESHOLD", "high").lower()
IMAGE_CACHE_DIR = os.path.join(_REPO_ROOT, "work_dirs", "vg_image_cache")

_client = None
_client_lock = threading.Lock()
_img_lock = threading.Lock()


def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                             base_url=os.getenv("OPENAI_BASE_URL"))
    return _client


# --------------------------------------------------------------------------- #
def _local_image_path(image_id, url):
    import requests
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    path = os.path.join(IMAGE_CACHE_DIR, f"{image_id}.jpg")
    with _img_lock:
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
    return path


def _b64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are a meticulous visual-grounding annotator. Identify the span(s) of the
LAST assistant answer that are HALLUCINATIONS — concrete visual claims that are
false or unsupported given the IMAGE.

You receive TWO images of the same scene: (1) the original, and (2) the same
image with colored bounding boxes drawn around scene-graph objects that are
relevant to the claims in the answer. You also receive a TEXT block with:
  - sg objects relevant to the answer (xyxy bbox + name + attributes),
  - per-claim sg verdicts. Important fields per claim:
      polarity = "affirm" | "deny"  (does the answer assert or deny the claim?)
      primary_status = "present_in_sg" | "absent_from_sg" | "unknown"
        (status of the SUBJECT entity, the first one listed)
      per_entity = each noun phrase + whether it matched the sg
      attr_verdict = attr_match | attr_mismatch | entity_absent | null
The scene graph is PARTIAL — it may miss objects that are actually visible.
If the image plainly shows something the sg doesn't list, trust the IMAGE.
If the sg says an attribute (e.g. "orange") that contradicts the answer
("green and white"), that is strong evidence the answer is hallucinated.

DENIALS:
  If polarity = "deny" (the answer says "no, there is no X" or "i cannot see Y"),
  flag ONLY when the denial is INCORRECT (the thing actually IS in the image).
  A correct denial — including refusing a false adversarial premise, or admitting
  an earlier mistake — is NOT a hallucination. Default to NOT flagging denials,
  and especially do not flag a denial whose primary entity is absent_from_sg.

Annotation rules (tight, single-span convention):
  1. Each span MUST be a contiguous, VERBATIM substring of the answer.
  2. Return the MINIMAL substring capturing the false claim:
       - absent object  -> short noun phrase ("the dog", "a notebook"),
       - existence claim-> short existence clause ("there is a traffic cone"),
       - attribute/relation -> short claim ("the parking meter is green and white").
  3. STRIP leading "yes," / "no," / lead-ins ("in the image, ", "i apologize...").
  4. STRIP trailing explanatory / justifying / speculative clauses.
  5. One span per DISTINCT false fact. A claim + its elaboration = one span.
  6. DO NOT flag plausibly-true claims, speculative / interpretive / advisory
     statements, generic commentary, hedging, or correct denials.

CONFIDENCE per span (mandatory):
  Rate each candidate hallucination with a confidence:
    "high"   — you can DIRECTLY SEE the contradicting evidence in the image
               (or the entity the answer asserts is plainly NOT in the image);
               the sg also supports the contradiction OR is silent but the
               image is decisive. You would bet on this being a hallucination.
    "medium" — likely a hallucination but you are not certain; the image is
               somewhat ambiguous, or the sg disagrees only weakly.
    "low"    — the claim is plausibly true given the image; you are unsure.
               (Default to "low" when in doubt. These are usually dropped.)
  When the answer gives a confidently specific detail that is NOT verifiable
  from the image and the image neither shows nor rules it out, prefer "medium"
  over "high". When the claim is a plausible / vague description that could
  reasonably match what is visible ("a passage for pedestrians or boats",
  "being used to play the game", "docked"), prefer "low".

  Two extra calibration rules — do NOT flag at high:
    (a) NEAR-SYNONYMS / category words. If the answer uses a broader or
        adjacent term that still names the same visible thing (computer vs
        console, vehicle vs car, bike vs bicycle, sneakers vs shoes,
        building vs structure), this is NOT a hallucination — at most "low".
    (b) GT-ONLY contradictions. If the only evidence the answer is wrong
        comes from the gt note (the image does not visibly contradict it),
        prefer "medium" — gt is a hint, but the image is the primary
        evidence. Use "high" only when the image itself clearly contradicts
        the claim (or the entity is plainly absent), not because gt says so.

Question-type guidance:
  - regular: most are grounded; flag only clear visual contradictions.
  - follow-up: flag a re-asserted false detail; correct retractions are not.
  - adversarial: if the answer affirms a thing clearly absent, flag the
    affirming claim only. Correct denials -> empty list.
  - unanswerable: if the answer gives a confident specific value for a detail
    not determinable from the image, flag the fabricated claim.

Return ONLY JSON of the form:
{"hallucinations": [
   {"span": "<verbatim substring of the answer>",
    "confidence": "high" | "medium" | "low",
    "reason": "<one short sentence: what the image/sg actually shows>"},
   ...
]}
Return {"hallucinations": []} if nothing is hallucinated.
"""


def _build_text_block(sg_verdicts, gt, history):
    lines = []
    if history:
        lines.append("Conversation so far (read-only context):")
        for h in history:
            lines.append(f"  [{h.get('q_type','?')}] Q: {h.get('prompt','')}")
            lines.append(f"             A: {h.get('response','')}")
    else:
        lines.append("Conversation so far: (this is the first turn)")
    if gt:
        lines.append(f"\nGround-truth note about the image (trusted): {gt}")
    lines.append("\nScene-graph objects matched to the claims in the answer "
                 "(xyxy = [x1,y1,x2,y2]; sg may be incomplete):")
    seen = set()
    any_box = False
    for v in sg_verdicts:
        for b in v.get("bboxes", []):
            key = (tuple(b["xyxy"]), tuple(b.get("names") or []))
            if key in seen:
                continue
            seen.add(key)
            attrs = ", ".join(b.get("attributes") or []) or "—"
            names = "/".join(b.get("names") or [b.get("entity", "")])
            lines.append(f"  - {names}  xyxy={b['xyxy']}  attrs=[{attrs}]")
            any_box = True
    if not any_box:
        lines.append("  (no sg objects matched any noun phrase in the answer)")
    lines.append("\nPer-claim sg verdicts:")
    if not sg_verdicts:
        lines.append("  (no atomic visual claims detected in the answer)")
    for v in sg_verdicts:
        parts = [f"primary={v.get('primary_status','?')}"]
        if v.get("attr_verdict"):
            parts.append(f"attr={v['attr_verdict']}")
        if v.get("rel_present") is not None:
            parts.append(f"rel_in_sg={v['rel_present']}")
        per_e = ", ".join(
            f"{pe['entity']}={'sg' if pe['in_sg'] else 'NOT-sg'}"
            for pe in (v.get("per_entity") or [])
        ) or "—"
        lines.append(f"  - [{v.get('kind','?')}/{v.get('polarity','?')}] "
                     f"\"{v['claim_text']}\"  -> {' '.join(parts)}  entities=[{per_e}]")
    return "\n".join(lines)


def _build_messages(orig_b64, ann_b64, text_block, prompt, response, q_type):
    user_text = f"""{text_block}

================ REAL ITEM TO ANNOTATE ================
Question type: {q_type}
Q: {prompt}
A (annotate THIS answer only): {response}

The first image is the ORIGINAL. The second image has bounding boxes drawn
around the sg-matched relevant objects above (the labels show the sg names +
attributes).

Output the hallucinated substrings of A as JSON:
{{"hallucinations": [...]}}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ann_b64}"}},
        ]},
    ]


# --------------------------------------------------------------------------- #
# Span snapping (same idea as v1)
# --------------------------------------------------------------------------- #
def _snap(span, response):
    if not span or not span.strip():
        return None
    if span in response:
        return span
    low_r, low_s = response.lower(), span.lower()
    i = low_r.find(low_s)
    if i >= 0:
        return response[i:i + len(span)]
    m = difflib.SequenceMatcher(None, low_s, low_r, autojunk=False).find_longest_match(
        0, len(low_s), 0, len(low_r))
    if m.size >= max(8, int(0.6 * len(low_s))):
        return response[m.b:m.b + m.size].strip()
    return None


def _postprocess(raw, response):
    """Return list of {span, confidence, reason} with each span snapped to an
    exact substring of the response. Drops anything that can't snap."""
    out, seen = [], set()
    for item in raw:
        if isinstance(item, str):
            span, conf, reason = item, "high", ""
        elif isinstance(item, dict):
            span = item.get("span") or item.get("hallucination") or ""
            conf = (item.get("confidence") or "high").strip().lower()
            reason = (item.get("reason") or "").strip()
        else:
            continue
        snapped = _snap(span.strip(), response) if span else None
        if not snapped or snapped in seen:
            continue
        seen.add(snapped)
        if conf not in _CONF_ORDER:
            conf = "high"
        out.append({"span": snapped, "confidence": conf, "reason": reason})
    return out


def _parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    blob = m.group(0) if m else text
    obj = json.loads(blob)
    spans = obj.get("hallucinations", obj.get("hallucination", []))
    if isinstance(spans, str):
        spans = [{"span": spans, "confidence": "high", "reason": ""}]
    return spans or []


def _apply_threshold(candidates, threshold):
    """Filter candidates by confidence >= threshold. Returns list of span strings."""
    thr = _CONF_ORDER.get(threshold, _CONF_ORDER["high"])
    return [c["span"] for c in candidates
            if _CONF_ORDER.get(c.get("confidence", "high"), 0) >= thr]


# --------------------------------------------------------------------------- #
def _cache_key(image_id, round_id, response, q_type, gt, sg_signature):
    h = hashlib.sha1()
    h.update(json.dumps(
        [PROMPT_VERSION, MODEL, REASONING_EFFORT, image_id, round_id,
         response, q_type, gt or "", sg_signature],
        ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def _cache_get(key):
    p = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None


def _cache_put(key, value):
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(value, open(os.path.join(CACHE_DIR, key + ".json"), "w"))


def _sg_signature(sg):
    # short stable summary of the sg so cache invalidates if the sg changes
    return f"o={len(sg.get('objects') or {})},r={len(sg.get('relationships') or [])}"


# --------------------------------------------------------------------------- #
def annotate_round(image, prompt, response, q_type, history=None, gt=None,
                      round_id=None, use_cache=True, return_debug=False,
                      confidence_threshold=None):
    """``image`` is the per-image dict from the result file (has image_id, url,
    optionally sg). Returns a list of hallucination spans (exact substrings of
    response) at or above ``confidence_threshold`` (default env
    ``AGENTIC_CONFIDENCE_THRESHOLD`` = "high").

    The cache stores ALL candidates with their confidences so the threshold
    can be tuned offline without re-calling the API.
    """
    threshold = (confidence_threshold or CONFIDENCE_THRESHOLD).lower()
    if not response or not response.strip():
        return ([], {"reason": "empty response"}) if return_debug else []
    sg = get_sg(image)
    sig = _sg_signature(sg)
    key = _cache_key(image["image_id"], round_id, response, q_type, gt, sig)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            cands = cached.get("candidates")
            if cands is None:
                # legacy cache format: just spans, treat as "high"
                legacy = cached.get("spans", cached if isinstance(cached, list) else [])
                cands = [{"span": s, "confidence": "high", "reason": ""} for s in legacy]
            spans = _apply_threshold(cands, threshold)
            return (spans, cached.get("debug", {})) if return_debug else spans

    # Step A — claim decomposition
    claims = decompose(prompt, response)
    # Step B — sg lookup
    verdicts = evaluate_claims(claims, sg)
    # Step C — vision judge with original + annotated image
    img_path = _local_image_path(image["image_id"], image["url"])
    orig_b64 = _b64_image(img_path)
    all_boxes = []
    for v in verdicts:
        all_boxes.extend(v.get("bboxes", []))
    ann_b64 = annotate(img_path, all_boxes, max_boxes=12)
    text_block = _build_text_block(verdicts, gt, history)
    messages = _build_messages(orig_b64, ann_b64, text_block, prompt, response, q_type)

    last_err = None
    for attempt in range(5):
        try:
            resp = _get_client().chat.completions.create(
                model=MODEL, messages=messages,
                extra_body={"reasoning_effort": REASONING_EFFORT}
                if MODEL.lower().startswith(("gpt-5", "o1", "o3", "o4"))
                else {},
            )
            candidates = _postprocess(_parse_json(resp.choices[0].message.content), response)
            debug = {"claims": claims, "verdicts": verdicts, "candidates": candidates}
            if use_cache:
                _cache_put(key, {"candidates": candidates, "debug": debug})
            spans = _apply_threshold(candidates, threshold)
            return (spans, debug) if return_debug else spans
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    print(f"[annotate_round] giving up image {image['image_id']} round {round_id}: {last_err}")
    return ([], {"error": str(last_err)}) if return_debug else []


def to_human_format(spans):
    return [{"hallucination": s, "reason": ""} for s in spans]
