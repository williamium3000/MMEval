"""Agentic hallucination annotator.

Given ONE (question, answer) pair grounded in an image, predict which exact
substrings of the answer are hallucinated (not supported by the image), mirroring
the human annotation style in
``work_dirs/human/result/dyna-v18/*_first50.json``.

The unit of work is a single round: ``annotate_round`` is called once per
(prompt, response) pair. Prior turns are passed as *read-only context* (needed
because follow-up / adversarial questions reference earlier turns), but only the
single current answer is annotated.

Output format matches the human files::

    "hallucination": [ {"hallucination": "<exact substring of answer>", "reason": ""}, ... ]

An empty list means the answer is fully grounded (no hallucination).

Model: gpt-5 (vision) via the uniapi gateway, using the project ``.env`` config
(``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``). Swap ``AGENTIC_MODEL`` to point at a
locally vLLM-hosted VLM instead.
"""

import os
import re
import json
import base64
import hashlib
import difflib
import threading

import requests
from dotenv import load_dotenv
from openai import OpenAI

# Resolve .env relative to repo root (graders/agentic/ -> ../../).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

# Bump this whenever the prompt / decoding changes so the on-disk cache invalidates.
PROMPT_VERSION = "v5"

MODEL = os.environ.get("AGENTIC_MODEL", "gpt-5")
REASONING_EFFORT = os.environ.get("AGENTIC_REASONING_EFFORT", "low")
CACHE_DIR = os.environ.get(
    "AGENTIC_CACHE_DIR", os.path.join(os.path.dirname(__file__), ".cache")
)
IMAGE_CACHE_DIR = os.path.join(_REPO_ROOT, "work_dirs", "vg_image_cache")

_client = None
_client_lock = threading.Lock()
_img_lock = threading.Lock()


def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            _client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
    return _client


# --------------------------------------------------------------------------- #
# Image handling
# --------------------------------------------------------------------------- #
def _image_data_url(image_id, url):
    """Download (and cache) the VG image, return a base64 data URL."""
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    path = os.path.join(IMAGE_CACHE_DIR, f"{image_id}.jpg")
    with _img_lock:
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are a meticulous visual-grounding annotator. You are given an IMAGE, a short
multi-turn conversation about that image, and ONE answer (the LAST assistant turn)
produced by a vision-language model. Your job: identify the span(s) of THAT answer
that are HALLUCINATIONS — concrete claims that are false or unsupported given the
image.

A span is a HALLUCINATION when it makes a CONCRETE VISUAL claim that:
  - contradicts what is visible in the image, OR
  - asserts the existence of an object/person/text/relation that is NOT present, OR
  - gives a specific detail (color, count, breed, material, brand, action,
    position/relation) that cannot be confirmed from the image (it was made up).

Do NOT flag (these are never hallucinations):
  - claims that are actually TRUE / plausibly supported by the image,
  - hedging, refusals, apologies, "i cannot tell", "it is unclear",
  - speculative, interpretive, advisory, or purpose/benefit statements
    (e.g. "this suggests the area is well-connected", "which helps maintain order",
    "making it easy for pedestrians to access it", "it is important to pay attention
    to traffic signals"),
  - generic commentary or correct restatements of the question.

Be CONSERVATIVE about detection. Default to NOT flagging. Only flag a claim when
you can positively determine from the image that it is false or unsupported.
If a statement is plausibly consistent with the image, leave it unflagged.

GRANULARITY — match the human annotator's tight style:
  1. Every returned span MUST be copied VERBATIM as a contiguous exact substring of
     the answer (same words/order, lowercase as given). Never paraphrase.
  2. Return the MINIMAL substring that captures the false claim:
       - if the hallucination is an OBJECT/PERSON that is simply not in the image,
         mark just the short noun phrase naming it ("the dog", "a notebook",
         "coffee mug", "the computer tower");
       - if it is an EXISTENCE assertion, mark the short existence clause
         ("there is a traffic cone", "there is a bus stop nearby along the sidewalk");
       - if it is an ATTRIBUTE or RELATION, mark the short claim
         ("the parking meter is green and white", "directly in front of the white van").
  3. STRIP the leading "yes," / "no," and conversational lead-ins ("in the image, ",
     "i apologize for the confusion. ").
  4. STRIP trailing explanatory / justifying / speculative clauses and sentences,
     and trailing scope words ("...where the bikes are parked", "...present near the
     crosswalk"). Flag ONLY the false claim itself, not the reasoning attached to it.
  5. If several DISTINCT false visual facts appear, return one span per fact.
     (A single false claim followed by elaboration = ONE span, the claim only.)
  6. If the answer is fully grounded / correct, return an empty list. In particular,
     a LONG descriptive answer about objects that ARE present (e.g. describing a desk
     setup, cords, a street scene) is usually grounded — do NOT flag plausible
     descriptive detail unless it clearly contradicts or is absent from the image.

Question-type guidance:
  - regular: a normal question; most answers are grounded. Flag only clear visual
    contradictions/fabrications. -> usually empty list.
  - follow-up: builds on an earlier turn. If it repeats/elaborates a previously
    hallucinated detail, flag the false detail (tight). A correct apology /
    retraction is NOT a hallucination.
  - adversarial: the question PRESUPPOSES an object/relation that may not exist. If
    the answer AFFIRMS a thing that is clearly absent from the image, flag the
    affirming claim only (e.g. "there is a bus stop nearby along the sidewalk"). If
    the answer correctly denies it, return empty list.
  - unanswerable: the question asks for a detail not determinable from the image
    (e.g. an attribute of an object that is not present). If the answer gives a
    confident specific value, flag the fabricated claim (tight, e.g.
    "the dog sitting next to the man is white").

Return ONLY a JSON object, no prose:
{"hallucinations": ["<verbatim substring>", ...]}
Return {"hallucinations": []} if nothing is hallucinated.
"""

# Text-only few-shot anchors granularity & format (drawn from real human
# annotations). The real item always carries its actual image; these carry a
# one-line scene gloss so they parse. Note the TIGHT spans: no "yes,", no tails.
FEWSHOT = """\
Below are reference examples (each with a one-line description of its image)
showing the expected TIGHT granularity. Learn the STYLE, then judge the real item
from its actual image.

[scene: a city sidewalk, two men talking, parked cars incl. a white van, an orange
parking meter, a crosswalk; NO dog, NO bench, NO bus stop, NO traffic cone]
Example A (regular, grounded -> nothing flagged)
Q: Is the white van parked near the orange parking meter?
A: yes, the white van is parked near the orange parking meter.
-> {"hallucinations": []}

Example B (regular, speculative tail not flagged)
Q: How wide does the road look?
A: the road appears quite wide, providing ample space for vehicles and pedestrians.
-> {"hallucinations": []}

Example C (follow-up, strip explanatory tail)
Q: Point out features of the parking meter that stand out.
A: the parking meter is green and white, which is an unusual color combination.
-> {"hallucinations": ["the parking meter is green and white"]}

Example D (adversarial, strip "yes," + tail; affirmed absent object)
Q: Is there a traffic cone present near the crosswalk?
A: yes, there is a traffic cone present near the crosswalk, which helps to guide pedestrians.
-> {"hallucinations": ["there is a traffic cone"]}

Example E (adversarial, two distinct false facts)
Q: Is there a bench on the sidewalk?
A: yes, there is a bench on the sidewalk. the image shows a man sitting on a bench, which is located near the two men standing on the sidewalk.
-> {"hallucinations": ["there is a bench on the sidewalk", "the image shows a man sitting on a bench"]}

Example F (unanswerable, fabricated attribute of absent object)
Q: What is the breed of the dog playing near the bike?
A: the dog playing near the bike is a pit bull.
-> {"hallucinations": ["the dog playing near the bike is a pit bull."]}
"""


def _build_messages(image_data_url, prompt, response, q_type, history, gt):
    ctx_lines = []
    if history:
        for h in history:
            ctx_lines.append(f"  [{h.get('q_type','?')}] Q: {h.get('prompt','')}")
            ctx_lines.append(f"            A: {h.get('response','')}")
    context_block = (
        "Conversation so far (read-only context):\n" + "\n".join(ctx_lines)
        if ctx_lines
        else "Conversation so far: (this is the first turn)"
    )
    gt_block = (
        f"\nGround-truth note about the image (trusted): {gt}\n"
        if gt
        else ""
    )
    user_text = f"""{FEWSHOT}

================ REAL ITEM TO ANNOTATE ================
{context_block}
{gt_block}
Question type of the turn to annotate: {q_type}
Q: {prompt}
A (annotate THIS answer only): {response}

List the hallucinated substrings of A as JSON: {{"hallucinations": [...]}}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


# --------------------------------------------------------------------------- #
# Span post-processing: snap each predicted span to an exact substring.
# --------------------------------------------------------------------------- #
def _snap_to_substring(span, response):
    """Return an exact substring of ``response`` matching ``span``, or None.

    Exact match wins; otherwise fall back to the longest contiguous matching
    block (handles minor model paraphrase / added words), keeping it only if it
    covers most of the predicted span.
    """
    if not span or not span.strip():
        return None
    if span in response:
        return span
    low_resp = response.lower()
    low_span = span.lower()
    idx = low_resp.find(low_span)
    if idx >= 0:
        return response[idx : idx + len(span)]
    # Longest contiguous matching block between span and response.
    sm = difflib.SequenceMatcher(None, low_span, low_resp, autojunk=False)
    m = sm.find_longest_match(0, len(low_span), 0, len(low_resp))
    if m.size >= max(8, int(0.6 * len(low_span))):
        return response[m.b : m.b + m.size].strip()
    return None


def _postprocess(raw_spans, response):
    out, seen = [], set()
    for s in raw_spans:
        if not isinstance(s, str):
            continue
        snapped = _snap_to_substring(s.strip(), response)
        if snapped and snapped not in seen:
            seen.add(snapped)
            out.append(snapped)
    return out


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def _cache_key(image_id, round_id, response, q_type, gt):
    h = hashlib.sha1()
    h.update(
        json.dumps(
            [PROMPT_VERSION, MODEL, REASONING_EFFORT, image_id, round_id,
             response, q_type, gt or ""],
            ensure_ascii=False,
        ).encode("utf-8")
    )
    return h.hexdigest()


def _cache_get(key):
    p = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _cache_put(key, value):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, key + ".json"), "w") as f:
        json.dump(value, f)


def _parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    blob = m.group(0) if m else text
    obj = json.loads(blob)
    spans = obj.get("hallucinations", obj.get("hallucination", []))
    if isinstance(spans, str):
        spans = [spans]
    return spans or []


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def annotate_round(image_id, url, prompt, response, q_type,
                   history=None, gt=None, round_id=None, use_cache=True):
    """Annotate ONE answer. Returns a list of exact-substring hallucination spans."""
    if not response or not response.strip():
        return []
    key = _cache_key(image_id, round_id, response, q_type, gt)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    image_data_url = _image_data_url(image_id, url)
    messages = _build_messages(image_data_url, prompt, response, q_type, history, gt)

    last_err = None
    for attempt in range(5):
        try:
            resp = _get_client().chat.completions.create(
                model=MODEL,
                messages=messages,
                extra_body={"reasoning_effort": REASONING_EFFORT}
                if MODEL.lower().startswith(("gpt-5", "o1", "o3", "o4"))
                else {},
            )
            content = resp.choices[0].message.content
            spans = _parse_json(content)
            result = _postprocess(spans, response)
            if use_cache:
                _cache_put(key, result)
            return result
        except Exception as e:  # noqa: BLE001
            last_err = e
            import time
            time.sleep(min(2 ** attempt, 30))
    print(f"[annotate_round] giving up image {image_id} round {round_id}: {last_err}")
    return []


def to_human_format(spans):
    """Wrap predicted spans in the human-file schema."""
    return [{"hallucination": s, "reason": ""} for s in spans]
