# checkmmhal — sanity check before rerunning mmhal on VG OS rows

**Goal:** before re-running mmhal on the 3 VG dirs, surface the **existing**
`mmhal_*.json` `avg_score` values vs **what the LaTeX table currently shows**
vs **how the new pipeline would summarize them**, so we can decide whether
the rerun is needed at all.

## Path → LaTeX row mapping

| Dir on disk | LaTeX row |
|---|---|
| `work_dirs/vg/caption/` | `caption` row |
| `work_dirs/vg/final_run_v18_gpt4o_completed/` | `CEDI` row (original v18) |
| `work_dirs/vg/ablation_baseline2_completed/` (single _completed) | `baseline` row |
| `work_dirs/vg/ablation_baseline2_completed_completed/` (double, the one you mentioned) | **transcripts only — no mmhal exists yet** |

The double-`_completed` dir has 4 model transcripts (llava / InternVL3 / Qwen2.5-VL / gemma-3) but **zero `mmhal_*.json` files anywhere in it.** The mmhal artifacts that do exist for "baseline" live under the single-`_completed` dir, and even there only `opera-llava-1.5` has them (n=32).

## What the new pipeline does with `avg_score`

`scripts/graders/run_all_metrics.sh` summary code (line ~406):

```python
_avg = mmhal['overall_metrics']['avg_score']
mmhal_score = _avg / 6.0 * 100.0
```

i.e. **mmhal % = avg_score / 6 × 100** (mmhal scores are 0–6 integers, 6 = perfect, so this maps to "% of max").

## Cross-check: existing mmhal_*.json vs LaTeX

### caption/ — all 7 OS models have mmhal (n=100 each)

| Model | tex_value | raw avg_score | new_pipeline_pct (avg/6×100) | inverted (1−avg/6)×100 |
|---|---|---|---|---|
| LLaVA-1.5 7B          | 48.0% | 3.1300 | **52.2%** | 47.8% |
| InternVL2 8B          | 49.0% | 3.4900 | **58.2%** | 41.8% |
| InternVL2.5 8B        | 44.0% | 3.4600 | **57.7%** | 42.3% |
| InternVL3 8B Instruct | 26.0% | 4.4400 | **74.0%** | 26.0% |
| Qwen2.5-VL-7B Instruct| 16.0% | 5.0900 | **84.8%** | 15.2% |
| gemma-3-12b-it        | 49.0% | 3.4700 | **57.8%** | 42.2% |
| opera-llava-1.5       | 57.0% | 2.8200 | **47.0%** | 53.0% |

### final_run_v18_gpt4o_completed/ — 6 of 7 (no opera) have mmhal (n=1873–2081 each, much larger than 100)

| Model | tex_value (CEDI row) | raw avg_score | new_pipeline_pct | inverted |
|---|---|---|---|---|
| LLaVA-1.5 7B          | 65.3% | 2.5762 | **42.9%** | 57.1% |
| InternVL2 8B          | 33.6% | 4.0196 | **67.0%** | 33.0% |
| InternVL2.5 8B        | 26.9% | 3.9354 | **65.6%** | 34.4% |
| InternVL3 8B Instruct | 35.5% | 3.7193 | **62.0%** | 38.0% |
| Qwen2.5-VL-7B Instruct| 24.2% | 4.5804 | **76.3%** | 23.7% |
| gemma-3-12b-it        | 41.9% | 3.6957 | **61.6%** | 38.4% |
| opera-llava-1.5       | —     | NOT FOUND | — | — |

### ablation_baseline2_completed*/ — only opera has mmhal

| Model | tex_value (baseline row) | raw avg_score | new_pipeline_pct |
|---|---|---|---|
| opera-llava-1.5 | — (currently `---`) | 2.5000 (n=32 only) | 41.7% |
| (all other 6 models) | — | NOT FOUND | — |

## What jumps out

The `new_pipeline_pct` and `tex_value` columns **diverge wildly** (e.g. Qwen2.5-VL caption: tex 16% vs new 84.8%), but the `inverted` column **roughly matches the tex value** for several rows (InternVL3: 26.0% ↔ 26.0%; Qwen2.5-VL: 16.0% ↔ 15.2%; LLaVA: 48.0% ↔ 47.8%).

**Interpretation:** the existing tex `mmhal` column appears to encode a **hallucination rate** (lower-mmhal-score = higher %), not an accuracy %. The arrow in the column header is `($\uparrow$)` (higher = better) but the values seem to be *(6−avg)/6 × 100*, which is "% hallucinated" — making the arrow misleading. The new pipeline's API rows (caption gpt-4o = 68.1% etc., already added by me) use the correct *avg/6 × 100* (accuracy %), so **the API rows and OS rows are NOT comparable** as currently displayed: a 68.1% in the API row means "68% of max" (good) while a 48.0% in the OS row means "48% hallucinated" (bad).

(Some rows like InternVL2 caption don't fit either formula cleanly — 49.0% vs 58.2%/41.8% — so I can't fully reverse-engineer without the original compute script. But the pattern is clearly "tex ≈ inverted of new pipeline" for most rows.)

## What this means for the rerun

Three options:

1. **Don't rerun. Just rewrite the OS rows of the tex** to use the *existing* `mmhal_*.json` values under the new-pipeline formula `avg/6×100`. Pros: no compute cost, all rows on same scale instantly. Cons: changes every existing OS mmhal cell (some go up, some go down dramatically — e.g. Qwen2.5-VL caption: 16% → 84.8%).

2. **Rerun mmhal** on all 3 dirs (only for our 7 models, only on records the new pipeline hasn't covered) — keeps the audit trail uniform. Pros: every cell is freshly computed by the same script that produced the API rows. Cons: ~hours of compute; final values will essentially equal `existing avg / 6 × 100` since the judges agree (per the 0/30-disagreement check earlier).

3. **Hybrid**: leave the existing OS tex values alone but flip the column header arrow / footnote to clarify "mmhal hallucination % (↓)" for OS rows and "mmhal accuracy % (↑)" for API rows. Cons: confusing, doesn't fix the comparability issue.

## My recommendation

**Option 1**: just refold the existing avg_scores into `avg/6×100` and overwrite the tex cells. Reasoning: we already verified Qwen-judge and gpt-4o-mini-judge produce 0/30 disagreements per sample, so the existing avg_scores are valid and rerunning would produce essentially the same numbers (saving many hours). The win is consistency: all rows would be on "% of max accuracy, ↑" scale and directly comparable.

If you want option 2 (full rerun), I'll back up `mmhal_*.json` files first per your instruction, then delete and let `run_all_metrics.sh --metrics mmhal --skip-existing` regenerate them — but only for the 7 models in our tex.

**Awaiting your call: 1, 2, or 3?**
