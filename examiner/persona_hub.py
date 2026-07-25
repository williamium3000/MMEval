"""PersonaHub (arXiv:2406.20094) persona sourcing + selection for the CEDI examiner.

Provenance of every prompt string in this file is marked inline:

  [PAPER]  = reproduced from the paper / their released repo, verbatim.
  [SCAFF]  = I/O scaffolding we had to add because the paper's figure prompts
             are prose-only ("The prompts shown in the figures throughout this
             paper are not exactly the prompt strings we used in our
             experiments; instead, they are simplified to fit the space" —
             §Appendix). Nothing here changes the paper's instruction, it only
             pins the output format so parse_json can read it.
  [NEW]    = written by us because CEDI has a constraint PersonaHub does not
             (image-grounding, first-person plausibility).

Reference implementation consulted: github.com/tencent-ailab/persona-hub
  code/prompt_templates.py, code/openai_synthesize.py.
"""

import hashlib
import json
import os
import random
import re

# ---------------------------------------------------------------------------
# 1. Persona sourcing
# ---------------------------------------------------------------------------

# The released 200k-persona preview of PERSONA HUB. openai_synthesize.py loads
# exactly this file and iterates it uniformly, one persona per synthesized datum.
PERSONA_HUB_REPO = "proj-persona/PersonaHub"
PERSONA_HUB_FILE = "persona.jsonl"


def load_persona_hub(path=None, max_personas=0):
    """Load PERSONA HUB personas as a list of strings.

    Mirrors persona-hub/code/openai_synthesize.py: read persona.jsonl, take the
    "persona" field, .strip() it.
    """
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(PERSONA_HUB_REPO, PERSONA_HUB_FILE, repo_type="dataset")
    personas = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            personas.append(json.loads(line)["persona"].strip())
            if max_personas and len(personas) >= max_personas:
                break
    return personas


# ---------------------------------------------------------------------------
# 2. Text-to-Persona  (paper §2.1, Figure 3)
# ---------------------------------------------------------------------------
# [PAPER] core instruction: "Who is likely to [read|write|like|dislike|...] the
#         text?" The bracket is a slot for the persona<->text relation and the
#         trailing "..." is the paper's own extension point.
# [PAPER] fine-grained requirement: the paper contrasts the coarse "a computer
#         scientist" with the fine-grained "a machine learning researcher
#         focused on neural network architectures and attention mechanisms".
# [SCAFF] the count + JSON output block.
# [NEW]   note 3. Empirically necessary: with the paper's original
#         read|write relations the model reads our context as a *dataset
#         artifact* and returns "a VQA dataset annotator", "a computer vision
#         researcher building street-scene datasets", "a location scout" for
#         nearly every VG/SVG scene. Those personas would push the examiner into
#         meta questions about the annotation, which directly violates CEDI's
#         "no disclosure of metadata / ask as if you are looking at the image"
#         rule. See RELATIONS_* below.

# The paper's original relation slot. Kept so the choice can be ablated.
RELATIONS_PAPER = "read|write|like|dislike"
# [NEW] Our default. Our "text" is not a document to be read, it is a scene to
# be inhabited in the first person, so the persona<->text relation that matters
# is presence in the scene, not readership of it.
RELATIONS_SCENE = "be in|act in|ask questions about"

TEXT_TO_PERSONA_PROMPT = """\
{text}

Who is likely to [{relations}] the text?

Note:

1. Give {n} distinct personas.
2. Each persona description should be fine-grained and specific rather than coarse-grained. For example, prefer "a machine learning researcher focused on neural network architectures and attention mechanisms" over "a computer scientist".
3. The persona must be someone who could plausibly be standing in this scene in person and speaking about what is in front of them. Do NOT return personas defined by their relationship to the text as a document or a dataset (e.g. dataset annotators, computer vision researchers, prompt writers, location scouts, caption authors) — this text describes a real place someone is standing in, not a document someone is working on.
4. Output format (STRICT):
```json
["persona 1", "persona 2", "..."]
```
You MUST only respond in the format described above. DO NOT RESPOND WITH ANYTHING ELSE.
"""


# ---------------------------------------------------------------------------
# 3. Persona-to-Persona  (paper §2.2, Figure 5)
# ---------------------------------------------------------------------------
# [PAPER] core instruction: "Who is in close relationship with the given
#         persona?" The paper runs six iterations of this expansion; we expose
#         `iterations` and default to 1 (see PersonaSelector).
# [SCAFF] count + JSON output block.

PERSONA_TO_PERSONA_PROMPT = """\
{persona}

Who is in close relationship with the given persona?

Note:

1. Give {n} distinct personas.
2. Each persona description should be fine-grained and specific rather than coarse-grained. For example, prefer "a machine learning researcher focused on neural network architectures and attention mechanisms" over "a computer scientist".
3. Output format (STRICT):
```json
["persona 1", "persona 2", "..."]
```
You MUST only respond in the format described above. DO NOT RESPOND WITH ANYTHING ELSE.
"""


# ---------------------------------------------------------------------------
# 4. Persona-driven synthesis note  (paper §3, code/prompt_templates.py)
# ---------------------------------------------------------------------------
# Every template in their prompt_templates.py carries the same conditioning
# clause. Reproduced verbatim with only the noun substituted
# ("math problem" -> "question"), which is the substitution they themselves make
# across math_template / instruction_template / knowledge_template / npc_template:
#
#   "You should make full use of the persona description to create the math
#    problem to ensure that the math problem is unique and specific to the
#    persona."
#
# Item 3 is [NEW]: PersonaHub's synthesis is unconstrained (a persona may force
# an arbitrarily exotic math problem), but a CEDI question must stay grounded in
# the image and answerable from it, so persona may only shape *how* the question
# is asked, never *what is true* about the scene.

PERSONA_CONDITIONING_NOTE = """\

**You are asking as the following persona:**
{persona}

Note:

1. You should make full use of the persona description to create the question to ensure that the question is unique and specific to the persona.
2. The persona should shape the *wording, focus, priorities and register* of your question — what this person would care about in this scene and how they would phrase it.
3. The persona must NOT change the facts of the image or the ground-truth answer. Stay grounded in the image content and the given context; do not invent expertise-specific objects that are not there, and do not let the persona push you into a question whose answer is not determinable from the image.
"""


# [NEW] Persona block for the conversation system prompt. CEDI's system prompt
# already asks for a "natural, open-ended, human-like conversation"; this names
# *which* human, which is the whole point of persona conditioning.
PERSONA_SYSTEM_BLOCK = """\

Persona:
You are conducting this conversation as the following person. Speak, prioritise and phrase your questions the way they would.
{persona}
"""


# ---------------------------------------------------------------------------
# 5. Deduplication  (paper §2.3)
# ---------------------------------------------------------------------------
# MinHash: 1-gram features, signature size 128, similarity threshold 0.9.
# Embedding: text-embedding-3-small, cosine similarity threshold 0.9.

MINHASH_NUM_PERM = 128
MINHASH_THRESHOLD = 0.9
EMBED_MODEL = "text-embedding-3-small"
EMBED_THRESHOLD = 0.9

_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _shingles(text):
    """1-gram features, per the paper's 'n-gram features (n=1)'."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def minhash_signature(text, num_perm=MINHASH_NUM_PERM, seed=0):
    rng = random.Random(seed)
    params = [(rng.randint(1, _MERSENNE_PRIME - 1), rng.randint(0, _MERSENNE_PRIME - 1))
              for _ in range(num_perm)]
    grams = _shingles(text)
    if not grams:
        return tuple([_MAX_HASH] * num_perm)
    base = [int(hashlib.sha1(g.encode()).hexdigest()[:8], 16) for g in grams]
    return tuple(min(((a * h + b) % _MERSENNE_PRIME) & _MAX_HASH for h in base)
                 for a, b in params)


def minhash_jaccard(sig_a, sig_b):
    return sum(1 for x, y in zip(sig_a, sig_b) if x == y) / len(sig_a)


class MinHashDeduper:
    """Greedy MinHash dedup at the paper's 0.9 threshold."""

    def __init__(self, threshold=MINHASH_THRESHOLD, num_perm=MINHASH_NUM_PERM):
        self.threshold = threshold
        self.num_perm = num_perm
        self.signatures = []

    def is_duplicate(self, text):
        sig = minhash_signature(text, self.num_perm)
        for known in self.signatures:
            if minhash_jaccard(sig, known) >= self.threshold:
                return True
        return False

    def add(self, text):
        self.signatures.append(minhash_signature(text, self.num_perm))

    def filter(self, texts):
        kept = []
        for t in texts:
            if not self.is_duplicate(t):
                self.add(t)
                kept.append(t)
        return kept


def embed_texts(texts, model=EMBED_MODEL):
    """OpenAI text-embedding-3-small, as used by the paper's embedding dedup."""
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-12)


# ---------------------------------------------------------------------------
# 6. Selection
# ---------------------------------------------------------------------------

class PersonaSelector:
    """Pick one persona per (image, context) conversation.

    Modes
    -----
    none           persona conditioning off (== plain v19d9; the ablation arm).
    hub            uniform draw from the released 200k PERSONA HUB. This is
                   PersonaHub's own sampling: openai_synthesize.py walks
                   persona.jsonl and pairs one persona with one synthesized
                   datum, no weighting, no filtering.
    text2persona   run their Text-to-Persona on the CEDI context text
                   (background + goal + relevant object names), then draw one
                   of the returned personas. This is the "context-based
                   selection" arm and the default for VG/SVG.
    hub_retrieval  embed the context and the hub with text-embedding-3-small,
                   keep the top-k nearest hub personas, draw one. Context-based
                   selection *over the released hub* rather than over freshly
                   generated personas.
    """

    def __init__(self, mode="text2persona", llm_chat=None, persona_path=None,
                 num_candidates=5, expand_iterations=0, expand_ratio=0.5,
                 dedup="minhash", retrieval_pool=20000, retrieval_topk=50,
                 relations=RELATIONS_SCENE, seed=0):
        self.mode = mode
        self.llm_chat = llm_chat
        self.relations = relations
        self.num_candidates = num_candidates
        self.expand_iterations = expand_iterations
        self.expand_ratio = expand_ratio
        self.dedup = dedup
        self.retrieval_topk = retrieval_topk
        self.rng = random.Random(seed)
        self.deduper = MinHashDeduper() if dedup == "minhash" else None
        self._emb_cache = {}

        self.personas = []
        if mode in ("hub", "hub_retrieval"):
            self.personas = load_persona_hub(
                persona_path,
                max_personas=retrieval_pool if mode == "hub_retrieval" else 0,
            )
            print(f"[persona] loaded {len(self.personas)} personas from PERSONA HUB")
        if mode == "hub_retrieval":
            self._hub_emb = embed_texts_batched(self.personas)

    # -- context -> text ----------------------------------------------------
    @staticmethod
    def context_to_text(context, object_names=None):
        """Flatten a CEDI context into the 'text' slot of Text-to-Persona.

        [NEW] PersonaHub feeds raw web text here. Our text is a first-person
        scene description, so we hand over exactly what the examiner itself is
        conditioned on: background, goal, and the relevant object names. Bounding
        boxes / object ids are deliberately excluded — the persona is meant to
        be a person in the scene, not an annotator of it.
        """
        parts = [f"Background: {context.get('background', '')}",
                 f"Goal: {context.get('goal', '')}"]
        if object_names:
            parts.append("Things visible in the scene: " + ", ".join(str(o) for o in object_names))
        return "\n".join(parts)

    # -- paper methods ------------------------------------------------------
    def text_to_persona(self, text, n=None):
        n = n or self.num_candidates
        prompt = TEXT_TO_PERSONA_PROMPT.format(text=text, n=n, relations=self.relations)
        out = self._chat_list(prompt)
        return out

    def persona_to_persona(self, persona, n=2):
        prompt = PERSONA_TO_PERSONA_PROMPT.format(persona=persona, n=n)
        return self._chat_list(prompt)

    def _chat_list(self, prompt):
        from utils.llm import parse_json
        for _ in range(3):
            out = self.llm_chat.chat([{"role": "user", "content": prompt}], parse_json)
            if isinstance(out, list) and out:
                return [str(p).strip() for p in out if str(p).strip()]
            if isinstance(out, dict):
                for v in out.values():
                    if isinstance(v, list) and v:
                        return [str(p).strip() for p in v if str(p).strip()]
        return []

    # -- main entry ---------------------------------------------------------
    def select(self, context, object_names=None):
        """Return {"persona": str, "mode": str, "candidates": [...]} or None."""
        if self.mode == "none":
            return None

        if self.mode == "hub":
            persona = self.rng.choice(self.personas)
            return {"persona": persona, "mode": "hub", "candidates": [persona]}

        if self.mode == "hub_retrieval":
            text = self.context_to_text(context, object_names)
            q = embed_texts([text])[0]
            scored = sorted(
                ((_cosine(q, e), p) for e, p in zip(self._hub_emb, self.personas)),
                key=lambda t: -t[0],
            )[: self.retrieval_topk]
            cands = [p for _, p in scored]
            return {"persona": self.rng.choice(cands), "mode": "hub_retrieval",
                    "candidates": cands[:10]}

        # text2persona (default)
        text = self.context_to_text(context, object_names)
        cands = self.text_to_persona(text)

        # Persona-to-Persona expansion (paper §2.2). Off by default: on everyday
        # VG/SVG scenes the relationship hop tends to walk *away* from the people
        # actually present in the frame, which costs plausibility. Turn it on to
        # widen style coverage.
        for _ in range(self.expand_iterations):
            extra = []
            for p in cands[: max(1, int(len(cands) * self.expand_ratio))]:
                extra.extend(self.persona_to_persona(p, n=2))
            cands.extend(extra)

        if self.deduper is not None:
            cands = self.deduper.filter(cands)
        if not cands:
            return None
        return {"persona": self.rng.choice(cands), "mode": "text2persona",
                "candidates": cands}


def embed_texts_batched(texts, batch=512):
    out = []
    for i in range(0, len(texts), batch):
        out.extend(embed_texts(texts[i:i + batch]))
    return out
