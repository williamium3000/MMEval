# v19 vs v19ban2type — multi-grader VG comparison

This run reruns the v19 dynamic-conversation examiner on Visual Genome for three
candidate VLMs and compares against the existing v19ban2type baseline across
agentic / MMHal / VALOR / CHAIR / SG-DELCON graders.

Companion bundle (raw outputs + SVG hallucination-vs-progress plot) on Drive:
`drive:/context-eval-mllm_results_2026-05-30/` (URL printed in the run log).

## What v19ban2type changes vs v19

`examiner/dyna_conv_v19ban2type.py` is forked from `examiner/dyna_conv_v19.py`
and bans the two trap question types — adversarial (`q_type=3`) and
unanswerable (`q_type=4`). Conversations only contain regular + follow-up
turns; the banned budget is redistributed to follow-ups.

## Setup

| Component | Backbone |
|---|---|
| Examiner LLM (v19 rerun) | Azure gpt-5.4-2026-03-05 (via `utils.llm.LLMChat`) |
| Candidate VLMs | gemma-3-12b-it, llava-1.5-7b-hf, Qwen2.5-VL-7B-Instruct |
| Agentic claim decomposer | qwen3-30b-instruct @ `http://[fdbd:dc61:18:24::36]:30000/v1` |
| Agentic vision judge | Azure gpt-5.4-2026-03-05 |
| MMHal / VALOR / CHAIR | qwen3-30b-instruct |
| SG DELCON parsing | Azure gpt-5.4-2026-03-05 |

Samples: 25 per (side, model). v19ban2type's image set is a subset of v19's,
so the comparison is apples-to-apples on 41–44 images per model.

## Code added / changed in this session

| Path | Change |
|---|---|
| `scripts/dyna-v19/v19_vg.sh` | New v19 VG launcher; drops `--parallel` (dyna_conv_v19.py doesn't accept it) and uses each conda env's python directly rather than `conda activate`. |
| `grader/agentic/plot_progress.py` | Aggregates agentic per-round hallucination over 10 conversation-progress bins and plots a (b)-style figure for "no history" vs "with history". |
| `grader/agentic/compare_vg_v19_vs_ban.py` | Agentic-only VG comparison with q_type breakdown. |
| `scripts/compare_v19_vs_ban_full.py` | Full multi-grader aggregator (agentic / MMHal / VALOR obj-exist & rel-pos / CHAIR sentence-weighted / SG DELCON). |
| `grader/agentic/annotator.py` | (1) `_get_client` falls back to `AzureOpenAI` when `AZURE_OPENAI_API_KEY` is set. (2) `_local_image_path` accepts `image_id` either with or without a `.jpg` suffix, sanitizes embedded slashes, and falls back to `work_dirs/svg_<src>_cache/` before raising `ImageUnavailable`. (3) `annotate_round` skips rounds whose image can't be fetched instead of crashing the entire job. (4) Accepts `image_url` as an alias for `url` (the VG examiner outputs `image_url`). |
| `grader/valor/gpt_model.py` | `llm()` signature changed from `(prompt, stop, model)` to `(prompt, model=None, stop=...)` so the existing `llm(prompt, args.model)` calls in `evaluate_object_existence.py` etc. actually use `args.model` instead of silently using the hardcoded `gpt-5.4-2026-03-05`. Retry capped at 5 (was infinite). Falls back to `_DEFAULT_MODEL` (env-driven) when `model` is `None`/`"gpt-5"`. |
| `grader/chair/chair_dyna_vg.py` | `--by_qtype` path now forwards `model=args.model` through to `compute_chair_by_qtype` → `extract_negated_objects_from_api`. Previously every API call requested the default `Qwen3-30B-A3B-Instruct-2507` deployment and got HTTP 404 from the local Qwen3 vLLM (which serves `qwen3-30b-instruct`). |
| `data/filtered_object_synsets_final.json` | Symlink to `grader/chair/filtered_object_synsets_final.json`. `grader/sg/llm_parser.py` hardcodes the `data/...` path. |

## Running

```bash
# 1) Examiner rerun
bash scripts/dyna-v19/v19_vg.sh "gemma-3-12b-it=1,llava-1.5-7b-hf=2,Qwen2.5-VL-7B-Instruct=3"

# 2) Agentic grader for both sides
bash work_dirs/vg/_agentic_v19_vs_ban_launcher.sh

# 3) Other graders (valor / sg-delta_con / chair / mmhal) for both sides
bash work_dirs/vg/_graders_launcher.sh   # 6 parallel streams, one per (side, model)

# 4) Aggregate + print comparison table
PYTHONPATH=. python3 scripts/compare_v19_vs_ban_full.py

# 5) SVG (a different artifact: hallucination-vs-progress plot for v19 vs v19conv)
PYTHONPATH=. python3 -m grader.agentic.plot_progress \
    --no-history-dir   grader/agentic/predictions/v19_n50 \
    --with-history-dir grader/agentic/predictions/v19conv_n50 \
    --output           grader/agentic/predictions/plot_progress.png
```

## Result summary

(Δ = v19ban2type − v19; ↓/↑ marks the better direction.)

| Metric | gemma-3-12b-it | llava-1.5-7b-hf | Qwen2.5-VL-7B-Instruct |
|---|:---:|:---:|:---:|
| Agentic hall rate ↓ | +8.0 pp worse | −5.8 pp better | +3.0 pp worse |
| MMHal score ↑ | +0.06 better | +0.57 better | +0.29 better |
| MMHal hall rate ↓ | +0.2 pp ≈tie | −11.6 pp better | −4.8 pp better |
| VALOR obj-exist faithful_i ↑ | +0.061 better | +0.068 better | −0.015 ≈tie |
| VALOR obj-exist coverage_i ↑ | +0.073 better | +0.114 better | −0.003 ≈tie |
| VALOR rel-positional faithful_i ↑ | +0.034 better | +0.026 better | +0.041 better |
| CHAIR sentence hall rate ↓ | −8.8 pp better | −9.6 pp better | −3.0 pp better |
| CHAIRs (sentence-weighted) ↓ | −8.9 better | −9.7 better | −3.1 better |
| CHAIRi (sentence-weighted) ↓ | −1.24 better | −1.71 better | −0.18 better |
| CHAIR Cov_all ↑ | +3.2 better | +6.3 better | +6.2 better |
| SG DELCON distance ↓ | +0.038 ≈tie | +0.032 ≈tie | +0.163 worse |

Read: v19ban2type wins on 7 of 8 object-level graders, but the gain is mostly
mechanical — the two banned q_types are the highest-hallucination ones. Per
q_type, CHAIR rates inside `regular` and `follow-up` are within ~3 pp on both
sides. The agentic span-level grader disagrees for gemma and Qwen2.5-VL: the
budget redistributed to more follow-ups produces more hallucinated spans even
though each follow-up is slightly cleaner.

Machine-readable: `work_dirs/vg/_full_comparison.json`.

## Graders not run / skipped

| Grader | Reason |
|---|---|
| SG GED | Slow (`nx.graph_edit_distance` with 300 s timeout per pair). |
| VALOR `obj_attr` / `people_attr` / `rel_compar` | Script-level `KeyError` on Qwen3 output schema; not an env problem. |
| HaELM | Needs `llama-7b-hf` + `grader/HaELM/checkpoint`. |
| FaithScore | Needs `llava-v1.5-7b` checkpoint. |
