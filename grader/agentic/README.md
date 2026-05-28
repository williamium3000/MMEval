# Agentic hallucination grader

An LLM agent that reproduces the **human span-level hallucination annotation**
in `work_dirs/human/result/dyna-v18/*_first50.json`: given an image and one
`(question, answer)` turn, it marks the exact substrings of the answer that are
hallucinated (not supported by the image).

## What it does

For **one** `(prompt, response)` pair (the unit of work — see `annotate_round`),
three steps per round:

1. **Step A — claim decomposition** (text-only, cheap → local Qwen3-30B at
   `:8088` by default). `claims.py` splits the answer into atomic visual claims:
   `{claim_text, kind=exists|attribute|relation, entities, attribute, relation,
   polarity=affirm|deny}`. Speculative / advisory / interpretive sentences are
   skipped (they are not hallucinations under the human convention).
2. **Step B — scene-graph lookup** (deterministic Python, no API).
   `sg_lookup.py` matches each claim's noun phrases against `sg.objects` (name
   + light singular/lemma tolerance) and produces per-entity in-sg flags, a
   primary-entity status, an attribute-overlap verdict, and a relation-in-sg
   check. `sg_join.py` returns the sg from the input dict if present, else
   lazily loads it from `utils/vg` joining on `image_id` (covers files like
   the opera / llava human transcripts that don't bake sg in).
3. **Step C — vision judge** (`gpt-5` via the uniapi gateway). The judge
   receives TWO images — the original and a copy with **colored bboxes drawn**
   around sg-matched relevant objects (`image_annotate.py`) — plus a text
   block listing each bbox's xyxy + name + attributes, plus each claim's
   polarity + primary-entity status + per-entity in-sg flags + attr verdict.
   The judge is told the sg is PARTIAL and to trust the image when they
   disagree, with explicit denial / synonym / gt-deference rules. Output is a
   list of hallucinated substrings, each with a `confidence` ∈ {high, medium,
   low} and a one-sentence `reason`.

Prior turns are passed as *read-only context* (follow-up / adversarial turns
refer back to them) but only the single current answer is annotated. Output
matches the human schema:

```json
"hallucination": [{"hallucination": "<exact substring>", "reason": ""}, ...]
```

An empty list means the answer is fully grounded.

### Confidence threshold (FP filter)

The judge rates each candidate span; the threshold is applied at READ time so
the cache stays valid across threshold changes:

- `high` (default) — only certain hallucinations; drops "plausibly true" cases
  like `"there's a passage for pedestrians or boats"`, `"X is being used to
  play the game"`, broader-term synonyms (computer vs console).
- `medium` — also keep likely-but-uncertain calls.
- `low` — keep all candidates.

Set via `AGENTIC_CONFIDENCE_THRESHOLD=high|medium|low`. Re-running
`run_predict.py` after changing the threshold is free (cache hits).

## Files

| file | purpose |
|------|---------|
| `annotator.py`      | orchestrator (Step C) + on-disk cache + confidence filter |
| `claims.py`         | Step A — atomic-claim decomposer (local Qwen3) |
| `sg_join.py`        | scene graph: baked field or VG join on `image_id` |
| `sg_lookup.py`      | Step B — deterministic claim ↔ sg matcher |
| `image_annotate.py` | PIL bbox overlay |
| `run_predict.py`    | annotate a result file, write predictions (concurrent) |
| `evaluate.py`       | score vs human gold (detection + span overlap) |
| `dev_tune.py`       | run on a subset of the two gold files and print agreement |

## Run

```bash
PY=/raid/icy/iris/.conda/envs/mint-eval/bin/python   # any env with openai+dotenv+PIL

# Predict on a v18 result file (uses per-round gt grounding when present):
$PY -m grader.agentic.run_predict \
    --input  work_dirs/vg/final_run_v18_gpt4o_completed/InternVL3-8B-Instruct/InternVL3-8B-Instruct.json \
    --output grader/agentic/predictions/InternVL3-8B-Instruct.pred.json \
    --workers 12

# Score against a gold file:
$PY -m grader.agentic.evaluate --gold GOLD.json --pred PRED.json

# Dev loop on the two annotated files (cached by PROMPT_VERSION):
$PY -m grader.agentic.dev_tune --limit-images 15

# Re-threshold previously cached predictions (no new API calls):
AGENTIC_CONFIDENCE_THRESHOLD=medium $PY -m grader.agentic.run_predict ...
```

## Configuration (env vars)

| var | default | meaning |
|-----|---------|---------|
| `AGENTIC_MODEL` | `gpt-5` | vision judge model (point at a local vllm VLM to self-host) |
| `AGENTIC_REASONING_EFFORT` | `low` | reasoning effort for gpt-5 / o-series judges |
| `AGENTIC_CONFIDENCE_THRESHOLD` | `high` | minimum confidence to keep a span |
| `AGENTIC_CACHE_DIR` | `grader/agentic/.cache` | judge response cache |
| `AGENTIC_CLAIMS_BASE_URL` | `http://localhost:8088/v1` | claim-decomposer endpoint |
| `AGENTIC_CLAIMS_API_KEY` | `william` | bearer for the claim-decomposer endpoint |
| `AGENTIC_CLAIMS_MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | claim decomposer |

Vision judge auth uses the project `.env` (`OPENAI_API_KEY` / `OPENAI_BASE_URL`,
default uniapi).

## Metrics (`evaluate.py`)

- **Detection** (per round, and per q_type): precision / recall / F1 / accuracy of
  "did we flag ≥1 span" vs the human.
- **Span**: mean best char-IoU per gold span, match-rate @ IoU≥0.5, and char-token
  F1 / precision / recall over the union of spans.

The two human files use somewhat different span conventions (opera annotates
wider; llava is minimal and noisier); the held-out `_single` files line up with
the llava tight-span convention, which the prompt is tuned for. Round-level
detection is the stable, meaningful metric; span overlap is reported as a
secondary signal.
