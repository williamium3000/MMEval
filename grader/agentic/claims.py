"""Step A of the v2-agentic pipeline: decompose the answer into atomic visual
claims, using a cheap text-only LLM call (local Qwen3-30B at :8088 by default).

Each claim is::

    {"claim_text": "<verbatim or near-verbatim substring of answer>",
     "kind": "exists" | "attribute" | "relation" | "other",
     "entities": ["parking meter", "white van"],     # noun phrases mentioned
     "attribute": "green and white" | null,           # for kind=attribute
     "relation":  "next to" | null,                   # for kind=relation
     "polarity":  "affirm" | "deny"}

Only image-grounded visual claims are emitted. Generic / speculative / advisory
sentences are skipped (these are not hallucinations by the human convention).
"""

import os
import json
import re
import threading
import time

from openai import OpenAI

CLAIMS_BASE_URL = os.environ.get("AGENTIC_CLAIMS_BASE_URL", "http://localhost:8088/v1")
CLAIMS_API_KEY = os.environ.get("AGENTIC_CLAIMS_API_KEY", "william")
CLAIMS_MODEL = os.environ.get("AGENTIC_CLAIMS_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

_client = None
_lock = threading.Lock()


def _get_client():
    global _client
    with _lock:
        if _client is None:
            _client = OpenAI(base_url=CLAIMS_BASE_URL, api_key=CLAIMS_API_KEY)
    return _client


SYSTEM = """\
You decompose a vision-language-model answer into atomic VISUAL CLAIMS that can
be checked against the image. Each claim asserts something concrete about what
is visible: object existence, an object's attribute, or a relation between
objects. Skip generic, speculative, advisory, or interpretive sentences — they
are not visual claims (e.g. "this suggests the area is well-connected",
"it is important to pay attention to traffic signals", "this arrangement
ensures the van is parked legally"). Skip apologies and hedging.

For each claim, emit:
  - claim_text: the minimal substring of the answer expressing the claim
    (try to copy verbatim; if you must lightly normalize, keep the noun phrases
    intact so they can be string-matched later).
  - kind: "exists" | "attribute" | "relation" | "other"
  - entities: list of short noun phrases mentioned (lowercase, singular where
    natural, e.g. ["parking meter", "white van"]).
  - attribute: a single short attribute string, or null (only when kind=attribute).
  - relation: a short relation phrase, or null (only when kind=relation).
  - polarity: "affirm" if the answer asserts the fact, "deny" if it denies it.

Return ONLY JSON: {"claims": [ ... ]}. If the answer has no visual claims at
all (pure hedging / advice / apology), return {"claims": []}.
"""

_FEWSHOT = [
    {"role": "user", "content":
     'Q: Is there a bus stop nearby along the sidewalk?\n'
     'A: yes, there is a bus stop nearby along the sidewalk, and the area is well-connected.'},
    {"role": "assistant", "content":
     '{"claims":[{"claim_text":"there is a bus stop nearby along the sidewalk",'
     '"kind":"exists","entities":["bus stop","sidewalk"],"attribute":null,'
     '"relation":"nearby","polarity":"affirm"}]}'},
    {"role": "user", "content":
     'Q: Point out features of the parking meter.\n'
     'A: the parking meter is green and white, which is an unusual color combination.'},
    {"role": "assistant", "content":
     '{"claims":[{"claim_text":"the parking meter is green and white",'
     '"kind":"attribute","entities":["parking meter"],"attribute":"green and white",'
     '"relation":null,"polarity":"affirm"}]}'},
    {"role": "user", "content":
     'Q: What does the walk sign specify?\n'
     'A: the walk sign indicates it is safe to cross. it is important to pay attention to traffic signals.'},
    {"role": "assistant", "content": '{"claims":[]}'},
]


def _parse(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    blob = m.group(0) if m else text
    obj = json.loads(blob)
    cs = obj.get("claims", []) or []
    out = []
    for c in cs:
        if not isinstance(c, dict):
            continue
        out.append({
            "claim_text": (c.get("claim_text") or "").strip(),
            "kind":       (c.get("kind") or "other").strip().lower(),
            "entities":   [e.strip().lower() for e in (c.get("entities") or []) if isinstance(e, str) and e.strip()],
            "attribute":  c.get("attribute"),
            "relation":   c.get("relation"),
            "polarity":   (c.get("polarity") or "affirm").strip().lower(),
        })
    return out


def decompose(prompt, response):
    if not response or not response.strip():
        return []
    user = f"Q: {prompt}\nA: {response}"
    messages = [{"role": "system", "content": SYSTEM}, *_FEWSHOT,
                {"role": "user", "content": user}]
    last_err = None
    for attempt in range(4):
        try:
            r = _get_client().chat.completions.create(
                model=CLAIMS_MODEL, messages=messages, temperature=0,
                max_tokens=1200,
            )
            return _parse(r.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2 ** attempt, 20))
    print(f"[claims] giving up: {last_err}")
    return []
