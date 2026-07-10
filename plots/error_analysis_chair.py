"""CHAIR-based duplicate of Fig 11 (error_analysis.pdf).

Reuses the 3-panel layout from error_analysis.py but the per-turn
hallucination signal is CHAIRs (0/1) from grader/chair/chair_dyna_vg.py
--by_qtype output, which is per-sentence with round_id/q_type/CHAIRs.

Series:
  * "CHAIR -- no history"   : v18   evaluatee outputs
  * "CHAIR -- with history" : v18conv evaluatee outputs

Model overlap between v18 and v18conv is filtered to the non-InternVL
set (matching error_analysis.py's EXCLUDE_PREFIX filter for the MMHal
figure) and only models present in BOTH series are kept for a paired
comparison.

v18 signal is loaded from either:
  work_dirs/vg/final_run_v18_gpt4o_completed/{model}/{model}_chair_results.json
    (older per-model output with 2000+ per-turn sentences)
  work_dirs/vg/final_run_v18_gpt4o_completed/hallucinated_words_{model}.json
    (newer --by_qtype output produced during this session, top-level).

v18conv signal is loaded from:
  work_dirs/vg/final_run_v18_gpt4o_conv_completed/hallucinated_words_{model}.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from error_analysis import (
    Q_TYPES,
    REPO,
    rate_by_qtype,
    rate_by_progress,
    per_model_rates_by_qtype,
)
from error_analysis_pope_human import _plot_3panel


V18_DIR      = os.path.join(REPO, "work_dirs/vg/final_run_v18_gpt4o_completed")
V18CONV_DIR  = os.path.join(REPO, "work_dirs/vg/final_run_v18_gpt4o_conv_completed")
OUT_PDF      = os.path.join(REPO, "nips_paper/fig/error_analysis_chair.pdf")

EXCLUDE_PREFIX = ("InternVL", "internvl")


def _candidate_paths(base_dir, model):
    return [
        os.path.join(base_dir, model, f"{model}_chair_results.json"),
        os.path.join(base_dir, f"hallucinated_words_{model}.json"),
        os.path.join(base_dir, model, f"hallucinated_words_{model}.json"),
    ]


def _load_chair(base_dir, model):
    for p in _candidate_paths(base_dir, model):
        if not os.path.isfile(p):
            continue
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  skip {p}: {e}")
            continue
        sents = d.get("sentences", [])
        if not sents:
            continue
        # Require per-turn (round_id + q_type) shape — reject group-by-image files
        first = sents[0]
        if "round_id" not in first or "q_type" not in first:
            continue
        return p, sents
    return None, None


def _collect(base_dir, label):
    rows, used = [], []
    if not os.path.isdir(base_dir):
        return rows, used
    for model in sorted(os.listdir(base_dir)):
        if model.startswith(EXCLUDE_PREFIX):
            continue
        # Skip helper suffixes
        if any(s in model for s in ("_extracted", "_pope", "_summary", "_with_both")):
            continue
        # Also try top-level filename form
        if not (os.path.isdir(os.path.join(base_dir, model))
                or model.startswith("hallucinated_words_")):
            continue
        m = (model.replace("hallucinated_words_", "").replace(".json", "")
             if model.startswith("hallucinated_words_") else model)
        if m.startswith(EXCLUDE_PREFIX):
            continue
        p, sents = _load_chair(base_dir, m)
        if not sents:
            continue
        used.append(m)
        max_round_per_image = {}
        for s in sents:
            imid = s.get("image_id")
            r = int(s.get("round_id") or 0)
            if r > max_round_per_image.get(imid, 0):
                max_round_per_image[imid] = r
        for s in sents:
            qt = s.get("q_type")
            if qt not in Q_TYPES:
                continue
            r = int(s.get("round_id") or 0)
            mx = max_round_per_image.get(s.get("image_id"), 0) or 1
            h = bool(s.get("metrics", {}).get("CHAIRs"))
            rows.append((m, qt, r / mx, h))
    print(f"[{label}] {len(set(used))} models, {len(rows)} per-turn records")
    return rows, sorted(set(used))


def main():
    v18_rows,  v18_models  = _collect(V18_DIR,     "CHAIR v18")
    conv_rows, conv_models = _collect(V18CONV_DIR, "CHAIR v18conv")

    # Restrict both series to the intersection (paired comparison).
    keep = sorted(set(v18_models) & set(conv_models))
    if not keep:
        print("No overlapping non-InternVL evaluatees with per-turn CHAIR "
              "in both dirs — did the v18conv jobs finish?")
        print(f"  v18 has:     {v18_models}")
        print(f"  v18conv has: {conv_models}")
        return

    v18_rows  = [r for r in v18_rows  if r[0] in keep]
    conv_rows = [r for r in conv_rows if r[0] in keep]
    print(f"Paired evaluatees ({len(keep)}): {keep}")

    series = {
        "CHAIR -- no history":   v18_rows,
        "CHAIR -- with history": conv_rows,
    }

    _plot_3panel(
        series,
        out_pdf=OUT_PDF,
        source_label="CHAIR",
        pie_series_key="CHAIR -- no history",
        palette={
            "CHAIR -- no history":   "#D5E4F4",
            "CHAIR -- with history": "#E4F0E0",
        },
        edge={
            "CHAIR -- no history":   "#2F6FCC",
            "CHAIR -- with history": "#496F2C",
        },
        markers={
            "CHAIR -- no history":   "o",
            "CHAIR -- with history": "D",
        },
    )
    for k in series:
        s = " | ".join(
            f"{qt}={rate_by_qtype(series[k])[qt][0]*100:5.1f}% "
            f"(n={rate_by_qtype(series[k])[qt][1]})"
            for qt in Q_TYPES)
        print(f"  {k:26s}  {s}")


if __name__ == "__main__":
    main()
