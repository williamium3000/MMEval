# Agentic hallucination annotator

An LLM agent that reproduces the **human span-level hallucination annotation** in
`work_dirs/human/result/dyna-v18/*_first50.json`: given an image and one
`(question, answer)` turn, it marks the exact substrings of the answer that are
hallucinated (not supported by the image).

## What it does

For **one** `(prompt, response)` pair (the unit of work — see `annotate_round`):

1. Downloads + caches the Visual Genome image and sends it to a **vision LLM**
   (`gpt-5` via the uniapi gateway by default).
2. Passes the prior turns as *read-only context* (follow-up / adversarial turns
   refer back to them) but annotates **only the current answer**.
3. Returns a list of hallucinated substrings, post-processed so every span is an
   **exact substring** of the answer (paraphrases are snapped back, or dropped).
4. Output is written in the human-file schema:
   `"hallucination": [{"hallucination": "<span>", "reason": ""}, ...]`.
   An empty list means the answer is fully grounded.

## Files

| file | purpose |
|------|---------|
| `annotator.py`   | core agent: prompt, vision call, span snapping, on-disk cache |
| `run_predict.py` | run over a result file, write predictions (concurrent) |
| `evaluate.py`    | score predictions vs human gold (detection + span overlap) |
| `dev_tune.py`    | run on a subset of the two gold files and print agreement |

## Run

```bash
PY=/raid/icy/iris/.conda/envs/mint-eval/bin/python   # any env with openai+dotenv

# Predict on the held-out target file (uses per-round gt grounding when present):
$PY -m graders.agentic.run_predict \
    --input  work_dirs/vg/final_run_v18_gpt4o_completed/InternVL3-8B-Instruct/InternVL3-8B-Instruct.json \
    --output graders/agentic/predictions/InternVL3-8B-Instruct.pred.json \
    --workers 16

# Score against a gold file:
$PY -m graders.agentic.evaluate --gold GOLD.json --pred PRED.json

# Cheap iteration loop on the two annotated files (cached by PROMPT_VERSION):
$PY -m graders.agentic.dev_tune --limit-images 15
```

Model / decoding via env: `AGENTIC_MODEL` (default `gpt-5`),
`AGENTIC_REASONING_EFFORT` (default `low`), `AGENTIC_CACHE_DIR`. Point
`AGENTIC_MODEL` at a locally vLLM-hosted VLM to swap the backend. Responses are
cached on disk keyed by `(PROMPT_VERSION, model, image_id, round, response, ...)`,
so re-runs and prompt iterations only re-call what changed.

## How the prompt was tuned (only 1 q + 1 a fed per call)

I never showed the agent the gold annotation for the item it annotates. I fed it
one `(q, a)` at a time, compared its output to the human gold on the two available
files, and iterated the prompt:

- **v3** (generic "mark false claims"): over-flagged. Detection F1 opera 0.87 /
  llava 0.72, but llava span char-precision **0.12** — it marked whole sentences
  incl. "yes, ..." and trailing reasoning.
- Studying the gold revealed the human style is **tight**: spans drop the leading
  "yes,", drop trailing explanatory / speculative clauses, and *interpretive /
  advisory filler is never flagged*.
- **v4**: tight spans + conservative detection. llava span char-precision
  0.12 → 0.21, opera 0.76 → 0.84.
- **v5**: mark the **minimal noun phrase** when an object is simply absent
  (the gold often marks just `"the dog"` / `"coffee mug"` / `"notebook"`), and do
  **not** flag long *plausible* descriptions of objects that are present.

### Note: the two human files use *different* span conventions

- `opera-llava-1.5_cache` annotates wider and frequently **double-marks** a
  fabricated entity (the noun phrase **and** the whole sentence).
- `llava-1.5-7b-hf_single` annotates **minimally** (often a bare noun phrase) and
  is noisier (a few gold spans are copy-paste errors, e.g. `"there is a traffic
  light"` pasted onto a traffic-cone answer).

The held-out target `InternVL3-8B-Instruct_single` shares the **`_single`** suffix
with the llava file, so the prompt is biased toward the **tight, single-span
llava convention**. Because the gold itself is inconsistent and ultra-terse,
exact span boundaries are noisy to match; **round-level detection** (is this turn
hallucinated?) is the stable, meaningful metric, and **span overlap (IoU / char-F1)**
is reported as a secondary signal.

## v2-agentic — true multi-step grounding (`annotator_v2.py`)

`annotator.py` (v5) is a single LLM-as-judge call per round. `annotator_v2.py`
adds an actual agentic flow per round:

1. **Step A — claim decomposition** (text-only, cheap → local Qwen3-30B at
   `:8088`). `claims.py` splits the answer into atomic visual claims
   `{claim_text, kind=exists|attribute|relation, entities, attribute, relation,
   polarity=affirm|deny}`. Speculative / advisory sentences are dropped.
2. **Step B — scene-graph lookup** (deterministic Python, no API).
   `sg_lookup.py` matches each claim's noun phrases against `sg.objects` (name
   + light singular/lemma tolerance), produces per-entity in-sg flags, a
   primary-entity status, an attribute-overlap verdict, and a relation-in-sg
   check. `sg_join.py` returns the sg from the input dict if present, else
   lazily loads it from `utils/vg` (joining on `image_id`).
3. **Step C — vision judge** (`gpt-5`). The judge gets TWO images — the
   original and a copy with **colored bboxes drawn** around sg-matched
   relevant objects (`image_annotate.py`) — plus a text block with each
   bbox's xyxy + name + attributes, and each claim's polarity + primary-status
   + per-entity in-sg flags + attr verdict. The judge is told the sg is
   PARTIAL and to trust the image when they disagree. It is also given
   explicit DENIAL guidance — a correct "no, there is no X" is not a
   hallucination.

### v5 (one-shot) vs v2-agentic on the 5-image dev subset

| metric | OPERA v5 → v2 | LLAVA v5 → v2 |
|---|---|---|
| detection F1 | 0.866 → 0.886 | 0.714 → 0.737 |
| span mean IoU | 0.623 → 0.682 | 0.310 → 0.323 |
| match@IoU≥0.5 | 0.674 → **0.814** | 0.231 → 0.256 |
| char-token F1 | 0.651 → 0.675 | 0.327 → 0.312 |

Real but modest gains. The biggest is opera span-match@0.5 (+14pt) — the sg
bboxes let the judge anchor spans on the right object. v2 costs ~2× per round
(one cheap text call + one vision call) and runs ~2× slower.

Run:
```bash
$PY -m graders.agentic.run_predict_v2 \
    --input  work_dirs/vg/final_run_v18_gpt4o_completed/InternVL3-8B-Instruct/InternVL3-8B-Instruct.json \
    --output graders/agentic/predictions/InternVL3-8B-Instruct.v2.pred.json \
    --workers 12
```

A sample run trace + the annotated image the judge actually sees:
`graders/agentic/samples/image52_round4_*`.

## Metrics (`evaluate.py`)

- **Detection** (per round, and per q_type): precision / recall / F1 / accuracy of
  "did we flag ≥1 span" vs the human.
- **Span**: mean best char-IoU per gold span, match-rate @ IoU≥0.5, and char-token
  F1/precision/recall over the union of spans.
