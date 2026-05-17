# Rerun All CEDI Results on SVG-500 with gpt-5.4-mini Backbone

This doc briefs a **new agent** on a **new machine** (single Python env, no
conda available) to reproduce every CEDI cell in
`tmp/paper_results_index_rollup_with_sh.csv` using:

- **Examiner backbone**: `gpt-5.4-mini` (already hardcoded in
  `examiner/dyna_conv_v19d9.py:933` — *no patch needed*).
- **Original CEDI**: `examiner/dyna_conv_v19d9.py` unchanged (2 contexts per
  image — the v19conv default it inherits).
- **Content-ablation CEDI**: same `dyna_conv_v19d9.py` with the `CONTEXT_PROMPT`
  patched to generate **1 context per image** instead of 2 (the "1/5 context"
  ablation — see §6 for the exact diff).
- **Dataset**: SVG, **500 samples** (`--dataset svg --num_samples 500`). The
  full sweep is identical for both variants.

Companion file: **`tmp/rerun_cedi_gpt54_todo.csv`** — one row per
`(round × variant × model)` job, with a `status` column you flip
`todo → running → done` as you go.

---

## 1. Repo and branch

```bash
git clone https://github.com/williamium3000/context-eval-mllm.git
cd context-eval-mllm
git checkout develop-v2          # v19d9 + v19d9_vg.sh land here
git log --oneline -5             # sanity check; latest should mention v19.9
```

`develop-v2` is the canonical research branch. `main` is older / stable.

---

## 2. Why no `conda activate`?

Every script in `scripts/` calls `conda activate <env>` because the original
machine has 5 separate envs (one per model family). On a single-env box you
must:

- Delete or comment out the `eval "$(conda shell.bash hook)"` and
  `conda activate ${conda_env}` lines in any script you adapt.
- Install all deps into your single virtualenv.
- Run models in **rounds** — between rounds, swap the conflicting Python
  package versions (mainly `transformers`). See §3.

The launcher I'd hold up as the reference (and what to copy + simplify) is
`scripts/dyna-v19/v19d9_vg.sh`. Read it once, then ignore the conda/GPU-dispatch
plumbing and write a flat per-job invocation.

---

## 3. Round plan (5 rounds — one env at a time)

Each round shares a compatible `transformers` (and friends) version. The
companion TODO CSV lists exactly which models belong in which round; this
table is the human-readable summary.

| round | label | models | env hint |
|---|---|---|---|
| 1 | API only | `gpt-5.4-mini`, `gemini-2.5-flash` | `pip install 'openai>=1.30' huggingface_hub` (no GPU, no `torch` needed for examinee — only for the examiner LLM client; the API call is pure HTTP) |
| 2 | Qwen + LLaVA family | `Qwen2.5-VL-7B-Instruct`, `Qwen3-VL-8B-Instruct`, `llava-1.5-7b-hf` | `transformers>=4.45`, `qwen-vl-utils`, `accelerate`, `Pillow` |
| 3 | InternVL family | `InternVL2-8B`, `InternVL2_5-8B`, `InternVL3-8B-Instruct` | `transformers==4.40.0`, `timm`, `einops`, `sentencepiece`, `flash-attn==2.5.8 --no-build-isolation` |
| 4 | Gemma-3 | `gemma-3-12b-it` | `transformers>=4.50` (Gemma3 requires it), `accelerate`. Needs gated-repo access (`HF_TOKEN`). |
| 5 | OPERA-LLaVA | `opera-llava-1.5` | Custom — install per `infer/opera/` deps (`transformers==4.31`, `peft`, `bitsandbytes`). Checkpoint must be downloaded separately to `data/checkpoints/opera/llava-1.5`. |

**Why this order:** rounds 1 and 5 are fully independent of the GPU-model
env; rounds 2–4 are ordered roughly by how disruptive the `transformers`
version flip is (newer → older → newest). Doing round 1 first gives you a
quick smoke test that the examiner LLM (gpt-5.4-mini) is reachable before
you sink time into local model installs.

Inside a round, do **both variants for every model** before moving to the
next round (otherwise you'll swap envs twice).

---

## 4. Required env vars (set once per shell)

```bash
export OPENAI_API_KEY='<uniapi key>'
export OPENAI_BASE_URL='https://api.uniapi.io/v1'
export GEMINI_API_KEY="${OPENAI_API_KEY}"           # uniapi proxies gemini too
export GEMINI_API_BASE='https://api.uniapi.io/gemini'
export HF_TOKEN='<your hf token>'                   # round 4 needs this
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

export PYTHONPATH="./:infer:grader/easydetect"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/infer/LLaVA"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/infer/LLaVA/llava"
export PYTHONNOUSERSITE=1
```

The examiner LLM (`gpt-5.4-mini`) and the context-generator LLM (`gpt-5`)
are both invoked via the same uniapi gateway (`utils/llm.py` routes
`gpt-5*` and `o*` model names through `OPENAI_BASE_URL`).

---

## 5. Per-job invocation

```bash
python examiner/dyna_conv_v19d9.py \
  --dataset svg --num_samples 500 \
  --model_path '<HF id or local path>' \
  --outfile     work_dirs/svg500/v19d9-original_gpt54/<MODEL>.json \
  --cache_file  work_dirs/svg500/v19d9-original_gpt54/<MODEL>_cache.json \
  --max_rounds 20
```

Single-GPU execution: prepend `CUDA_VISIBLE_DEVICES=0` (or whichever GPU is
free). API examinees (round 1) need **no GPU**; set
`CUDA_VISIBLE_DEVICES=""` to avoid accidental allocation.

The cache file lets you resume after a crash — re-run the same command and
it picks up at the first un-cached sample.

The TODO CSV has a `run_command` column with a copy-pastable command for
every row.

---

## 6. The 1/5-context ablation patch

`v19d9` (inherited from v19conv) generates **2 contexts per image** by
default — the `CONTEXT_PROMPT` literally says *"Please generate two contexts
based on the image information."* (`examiner/dyna_conv_v19d9.py:91`) and
the rest of the file iterates over the returned list. For the **"1/5
context" ablation**, reduce the per-image context count from 5 to 1 by:

(a) editing the prompt to ask for 1 context, AND
(b) restricting the generation/iteration path to one context.

Concretely, **before running the ablation round**, apply this two-line
patch on a separate branch or a stash:

```bash
git stash                                    # save any in-progress edits
git checkout -b v19d9-1ctx
```

Then in `examiner/dyna_conv_v19d9.py`:

1. Replace `Please generate two contexts based on the image information.`
   (around line 91) with
   `Please generate one context based on the image information. Return a JSON list with a single dictionary.`
2. In `EvalSample.generate_context(...)` (around line 595), inside the
   `else:` branch (the no-`previous_context` path) where the prompt is
   formatted, **also** truncate the LLM output to a single dict:
   ```python
   contexts = self.llm_chat_context.chat(conversations, parse_json)
   if isinstance(contexts, dict):
       contexts = [contexts]
   contexts = contexts[:1]                  # NEW — keep only the first context
   ```
3. In `__call__` (around line 705), where it sees `len(prev_contexts) == 1`
   and asks for a second context, **skip** asking for a second by making
   the early-return path: if the cached / generated contexts already
   contain 1, do not generate another. (The current loop tries to top up to
   2; cap it at 1.)

Verify the patch with a 5-sample dry-run before launching the full
ablation:

```bash
python examiner/dyna_conv_v19d9.py --dataset svg --num_samples 5 \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --outfile /tmp/v19d9_1ctx_smoke.json \
  --cache_file /tmp/v19d9_1ctx_smoke_cache.json
```

Open `/tmp/v19d9_1ctx_smoke.json` and confirm each `image_id` appears
**exactly once** (one context = one conversation per image) — vs. the
original which has each `image_id` appear twice.

When you're done with the ablation sweep, `git checkout develop-v2` to
restore the 2-context original before running any original-variant jobs
on the same branch.

> **Open question worth raising with the user before you start**: the
> phrase "1/5 context" could mean
> (a) **1 generated context per image** (current interpretation — vs.
> v19d9's default 2 = halve, not 1/5), or
> (b) **1 ICL example in the context prompt** (v19d9 has 3 ICL examples;
> v18conv had 5). If they meant (b), instead delete two of the three ICL
> examples in `CONTEXT_PROMPT` (~line 41) and leave generation count
> at 2.
>
> Best to confirm before kicking off the ablation rounds.

---

## 7. SVG dataset (where the images come from)

`utils/svg.py` loads from a HuggingFace dataset. As of `develop-v2` it
expects `Icey444/svg5000_in_vg` (private — needs an HF token with read
access). If you see a 401 or 404 on `load_dataset`, the public mirror
is `nnonymous/svg5000_in_vg` (uploaded for the published code) — update
`utils/svg.py` to point at that.

500 samples × ~10–20 rounds/conv × 2 contexts ≈ 10k–20k examiner LLM
calls per *original* job, and ~5k–10k per *ablation* job. Plan API
spend accordingly (~$10–$30 per model run on gpt-5.4-mini at current
pricing).

---

## 8. Workflow per round

```
0. (once per round) install pip deps from §3 table
1. read the TODO CSV; filter status=todo AND round=<this round>
2. for each row in that filter:
     a. mark status=running, started_at=now, in the CSV
     b. paste run_command into the shell
     c. tail the log; wait for "Results saved to" line
     d. when finished, set status=done, finished_at=now
     e. if it crashes, leave status=running and add a note (see §9)
3. after all rows in the round are done, move to next round
```

Both **original and ablation jobs** for a given model live in the same
round (same env) — finish both before swapping envs.

---

## 9. Common failure modes (quick reference)

- **"openai.NotFoundError: gpt-5.4-mini not found"** — uniapi gateway
  returned 404. Re-check `OPENAI_BASE_URL` is the uniapi v1 base; do *not*
  use `api.openai.com` (that does not host gpt-5.4-mini).
- **Empty `response` strings + suspiciously short logs on API examinees** —
  hit when the examinee is a reasoning model (gpt-5*/o-series) and all
  tokens go to hidden reasoning. `utils/llm.py` already injects
  `reasoning_effort=minimal` for those families; if you see this on
  non-reasoning models, the patch may be missing in your branch — pull
  develop-v2 fresh.
- **`huggingface_hub.errors.GatedRepoError`** for `google/gemma-3-12b-it` —
  visit the model page on HF and accept the license, then re-export
  `HF_TOKEN`.
- **InternVL crashes with `RuntimeError: shapes do not match`** — version
  drift in `transformers`. Round 3 needs `transformers==4.40.0` exactly,
  not `>=4.40`.
- **Context-generation hangs at "asking regular question" forever** —
  this is the empty-`relevant_objects` bug fixed in `dyna_conv_v18conv.py`;
  v19d9 inherits the fix. If you see it, you're on an older copy — pull.
- **`_pope_dsg_v2.log` / DSG_v4 path** — DSG_v2/v4 is the POPE/CEDI yes-no
  verification pipeline. Not part of this CEDI sweep; do not run it
  unless the user separately asks for POPE/CEDI accuracy numbers.

---

## 10. Reporting back

When all 5 rounds × 2 variants × N models complete:

1. The TODO CSV should be entirely `status=done`.
2. Push the output JSONs to wherever the user designates (likely
   `gs://` or `hf://`, mirroring how `work_dirs/svg/` is shipped today).
3. Hand back two summary tables:
   - **Per-model CHAIRs / CHAIRi / Coverage / GED** from the new
     `all_metrics.json` files (run the grader pipeline — `scripts/grader/`
     is the reference; details TBD with the user).
   - **q_type distribution** (regular / follow-up / adversarial /
     unanswerable percentages per model). The whole point of switching to
     v19d9 is to verify the balance is now ~25/25/25/25, not the
     ~17/62/8/13 we saw on v19conv.

Save both tables alongside the JSONs, and post them in your final reply so
the user can compare against the pre-existing CSV without grepping disk.
