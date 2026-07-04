"""
Error analysis for §Results — produces a single 1x2 PDF for the paper:
  left  : hallucination rate by q_type
  right : hallucination rate vs % conversation progress

VG-only. Three experimental groups and two automatic graders are pooled into
four series:

  - CEDI standard,   MMHal       (light-blue, lighter shade)
  - CEDI standard,   POPE        (light-blue, darker shade)
  - CEDI long-ctx,   MMHal       (light-green)
  - Human            (sentence)  (warm color, footnoted)

"Standard" CEDI = every VG dyna run except the long-history examiner setting
(`final_run_v18_gpt4o_conv_completed`), which feeds the full multi-turn
history back to the examinee. POPE coverage is limited to 5 models, and is not
available on the long-history group, so that cell is left out (footnoted).

Per-turn hallucination signals:
  - MMHal : `has_hallucination` (bool) per turn.
  - POPE  : per-question yes/no, aggregated to per-turn = mean error rate over
            that turn's `dsg_qa` list, with the canonical GT flip applied for
            adversarial/unanswerable q_types (RESULT_FORMAT.md §5.3).
  - Human : turn is hallucinated iff `len(turn["hallucination"]) > 0`.
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

REPO = "/raid/william/project/context-eval-mllm"

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

# All VG dyna runs *except* the long-history one
CEDI_STD_DIRS = [
    "work_dirs/vg/final_run_v18_gpt4o_completed",
    "work_dirs/vg/v19_api_completed",
    "work_dirs/vg/v19",
    "work_dirs/vg/v20_api",
]
# Long-history run (examinee gets full conversation back as context)
CEDI_LONG_DIRS = [
    "work_dirs/vg/final_run_v18_gpt4o_conv_completed",
]
HUMAN_DIR = os.path.join(REPO, "reference/human annotation/context")

OUT_PDF = os.path.join(REPO, "plots/error_analysis.pdf")
OUT_CSV = os.path.join(REPO, "plots/error_analysis.csv")

Q_TYPES = ["regular", "follow-up", "adversarial", "unanswerable"]
N_BINS = 10  # 0-10%, 10-20%, ..., 90-100% of conversation progress


# ---------------------------------------------------------------------------
# MMHal collection (per-turn)
# ---------------------------------------------------------------------------

def _list_mmhal(dirs):
    files = []
    for d in dirs:
        full = os.path.join(REPO, d)
        if not os.path.isdir(full):
            continue
        for sub in sorted(os.listdir(full)):
            sub_path = os.path.join(full, sub)
            if not os.path.isdir(sub_path):
                continue
            for fn in os.listdir(sub_path):
                if (fn.startswith("mmhal_") and fn.endswith(".json")
                        and "results_summary" not in fn
                        and "with_both" not in fn
                        and "pope" not in fn
                        and "extracted" not in fn):
                    files.append(os.path.join(sub_path, fn))
    return files


def collect_mmhal(dirs, label):
    rows = []
    files = _list_mmhal(dirs)
    for p in files:
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  skip {p}: {e}")
            continue
        results = d.get("detailed_results", [])
        if not results:
            continue
        # model identity = the parent dir name (one mmhal file per model dir)
        model_key = os.path.basename(os.path.dirname(p))
        # progress = round_id / max round_id within the same record_index
        max_round = defaultdict(int)
        for r in results:
            max_round[r.get("record_index")] = max(
                max_round[r.get("record_index")], int(r.get("round_id") or 0))
        for r in results:
            qt = r.get("q_type")
            if qt not in Q_TYPES:
                continue
            rid = int(r.get("round_id") or 0)
            mx = max_round[r.get("record_index")] or 1
            rows.append((model_key, qt, rid / mx, bool(r.get("has_hallucination"))))
    print(f"[{label} mmhal] {len(files)} files, {len(rows)} per-turn records")
    return rows


# ---------------------------------------------------------------------------
# POPE collection (per-question -> per-turn)
# ---------------------------------------------------------------------------

POPE_FILES = [
    "work_dirs/vg/final_run_v18_gpt4o_completed/llava-1.5-7b-hf_with_both_answers.json",
    "work_dirs/vg/final_run_v18_gpt4o_completed/InternVL2_5-2B_with_both_answers.json",
    "work_dirs/vg/final_run_v18_gpt4o_completed/InternVL3-8B-Instruct_with_both_answers.json",
    "work_dirs/vg/final_run_v18_gpt4o_completed/Qwen2.5-VL-7B-Instruct_with_both_answers.json",
    "work_dirs/vg/v19_api/openai_gpt-4o_with_both_answers_isolated_all.json",
]


def _yn(s):
    if s is None:
        return None
    s = str(s).strip().lower()
    if s.startswith("y"):
        return "Yes"
    if s.startswith("n"):
        return "No"
    return None


def collect_pope():
    """Per-POPE-question error rate, tagged with parent turn's q_type and
    conversation progress.

    Canonical GT flip: for adversarial/unanswerable q_types, force GT="No"
    regardless of what was extracted from the transcript (RESULT_FORMAT.md §5.3).
    Emitting one row per POPE question (rather than collapsing to per-turn) so
    that the rate is comparable to standard POPE accuracy.
    """
    rows = []
    used = []
    for p in POPE_FILES:
        full = os.path.join(REPO, p)
        if not os.path.isfile(full):
            print(f"  skip POPE {p}: missing")
            continue
        try:
            d = json.load(open(full))
        except Exception as e:
            print(f"  skip POPE {p}: {e}")
            continue
        used.append(p)
        model_key = os.path.basename(p).split("_with_both_answers")[0]
        for s in d:
            convs = s.get("conversations", [])
            if not convs:
                continue
            mx = max((t.get("round_id") or 0) for t in convs) or 1
            for t in convs:
                qt = t.get("q_type")
                if qt not in Q_TYPES:
                    continue
                qa_list = t.get("dsg_qa") or []
                rid = int(t.get("round_id") or 0)
                prog = rid / mx
                for qa in qa_list:
                    if qt in ("adversarial", "unanswerable"):
                        gt = "No"
                    else:
                        gt = _yn(qa.get("gt_answer") or qa.get("answer"))
                    if gt is None:
                        continue
                    pred = qa.get("dynamic_response")
                    if pred is None:
                        continue
                    pred = _yn(pred) or pred
                    rows.append((model_key, qt, prog, pred != gt))
    print(f"[std pope] {len(used)} files, {len(rows)} per-question records")
    return rows


# ---------------------------------------------------------------------------
# Human annotation
# ---------------------------------------------------------------------------

def collect_human():
    rows = []
    files = sorted(f for f in os.listdir(HUMAN_DIR) if f.endswith(".json"))
    for fn in files:
        model_key = (fn.replace("_single_first50.json", "")
                       .replace("_cache_first50.json", ""))
        d = json.load(open(os.path.join(HUMAN_DIR, fn)))
        for s in d:
            convs = s.get("conversations", [])
            if not convs:
                continue
            mx = max((t.get("round_id") or 0) for t in convs) or 1
            for t in convs:
                qt = t.get("q_type")
                if qt not in Q_TYPES:
                    continue
                rid = int(t.get("round_id") or 0)
                rows.append((model_key, qt, rid / mx, bool(t.get("hallucination"))))
    print(f"[human] {len(files)} files, {len(rows)} per-turn records")
    return rows, files


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def rate_by_qtype(rows):
    out = {}
    for qt in Q_TYPES:
        sub = [h for (_m, q, _p, h) in rows if q == qt]
        out[qt] = (sum(sub) / len(sub) if sub else 0.0, len(sub))
    return out


def per_model_rates_by_qtype(rows):
    """Return {q_type: [per-model rate, ...]} for boxplot."""
    bag = defaultdict(lambda: defaultdict(list))  # qt -> model -> [hals]
    for (m, q, _p, h) in rows:
        if q in Q_TYPES:
            bag[q][m].append(h)
    out = {}
    for qt in Q_TYPES:
        out[qt] = [sum(v) / len(v) for v in bag[qt].values() if len(v) >= 5]
    return out


def rate_by_progress(rows, n_bins=N_BINS):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rates, counts = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            sub = [h for (_m, _q, p, h) in rows if lo <= p <= hi]
        else:
            sub = [h for (_m, _q, p, h) in rows if lo <= p < hi]
        rates.append(sum(sub) / len(sub) if sub else np.nan)
        counts.append(len(sub))
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, np.array(rates), counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Compare the SAME grader (MMHal) across two examiner settings -- v18
    # (CEDI without conversation history fed to the examinee) vs v18conv
    # (CEDI with full multi-turn history). Restricted to the 5 non-InternVL
    # evaluatees that have MMHal output in both directories.
    EXCLUDE_PREFIX = ("InternVL", "internvl")

    def filter_non_internvl(rows):
        return [r for r in rows if not r[0].startswith(EXCLUDE_PREFIX)]

    v18_dir  = ["work_dirs/vg/final_run_v18_gpt4o_completed"]
    conv_dir = ["work_dirs/vg/final_run_v18_gpt4o_conv_completed"]
    series = {
        "MMHal -- no history":   filter_non_internvl(collect_mmhal(v18_dir,  "v18")),
        "MMHal -- with history": filter_non_internvl(collect_mmhal(conv_dir, "v18conv")),
    }

    # ---- Aggregate ----
    agg_qt   = {k: rate_by_qtype(v)    for k, v in series.items()}
    agg_prog = {k: rate_by_progress(v) for k, v in series.items()}

    # ---- CSV dump ----
    with open(OUT_CSV, "w") as f:
        f.write("series,q_type,hallucination_rate,n\n")
        for k, byqt in agg_qt.items():
            for qt in Q_TYPES:
                r, n = byqt[qt]
                f.write(f"{k},{qt},{r:.4f},{n}\n")
        f.write("\nseries,bin_center,hallucination_rate,n\n")
        for k, (centers, rates, counts) in agg_prog.items():
            for c, r, n in zip(centers, rates, counts):
                f.write(f"{k},{c:.3f},{r:.4f},{n}\n")
    print(f"[csv] wrote {OUT_CSV}")
    for k in series:
        s = " | ".join(
            f"{qt}={agg_qt[k][qt][0]*100:5.1f}% (n={agg_qt[k][qt][1]})"
            for qt in Q_TYPES)
        print(f"  {k:25s}  {s}")

    # ---- Plot ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    palette = {
        "MMHal -- no history":   "#D5E4F4",  # light blue fill
        "MMHal -- with history": "#E4F0E0",  # light green fill
    }
    edge = {
        "MMHal -- no history":   "#2F6FCC",  # dark blue
        "MMHal -- with history": "#496F2C",  # dark green
    }
    markers = {
        "MMHal -- no history":   "o",
        "MMHal -- with history": "D",
    }
    keys = list(series.keys())

    fig, (axP, axL, axR) = plt.subplots(
        1, 3, figsize=(10.5, 2.4),
        gridspec_kw={"width_ratios": [1, 2, 2]},
    )

    # ---- Panel (a): pie chart of total hallucinated-turn counts by q_type ----
    # Uses the no-history CEDI series; each slice is the number of MMHal-flagged
    # hallucinated turns of that question type.
    pie_series_key = "MMHal -- no history"
    counts_by_qt = {qt: 0 for qt in Q_TYPES}
    for (_m, q, _p, h) in series[pie_series_key]:
        if h and q in counts_by_qt:
            counts_by_qt[q] += 1
    pie_counts = [counts_by_qt[qt] for qt in Q_TYPES]
    pie_colors = {
        "regular":      "#D5E4F4",
        "follow-up":    "#E4F0E0",
        "adversarial":  "#F0D0C8",
        "unanswerable": "#F6E4B8",
    }
    pie_edge = {
        "regular":      "#2F6FCC",
        "follow-up":    "#496F2C",
        "adversarial":  "#B87474",
        "unanswerable": "#B08830",
    }
    slices, texts, autotexts = axP.pie(
        pie_counts,
        labels=[qt.replace("-", "-\n") for qt in Q_TYPES],
        colors=[pie_colors[qt] for qt in Q_TYPES],
        wedgeprops=dict(edgecolor="white", linewidth=1.0),
        autopct="%1.0f%%",
        pctdistance=0.68,
        textprops=dict(fontsize=7),
    )
    for wedge, qt in zip(slices, Q_TYPES):
        wedge.set_edgecolor(pie_edge[qt])
        wedge.set_linewidth(0.8)
    for t in autotexts:
        t.set_fontsize(6.5)
    axP.set_title("(a) Hallucination count by question type")

    # ---- Panel (b): per-model boxplots by q_type ----
    box_keys = list(keys)
    box_data = {k: per_model_rates_by_qtype(series[k]) for k in box_keys}

    x = np.arange(len(Q_TYPES))
    w = 0.24
    offsets = np.linspace(-(len(box_keys) - 1) / 2,
                           (len(box_keys) - 1) / 2, len(box_keys)) * w
    for k, off in zip(box_keys, offsets):
        positions = x + off
        data = [[v * 100 for v in box_data[k][qt]] for qt in Q_TYPES]
        bp = axL.boxplot(
            data, positions=positions, widths=w * 0.9,
            patch_artist=True, manage_ticks=False,
            showfliers=False, whis=1.5,
            medianprops=dict(color="black", linewidth=1.0),
            boxprops=dict(facecolor=palette[k], edgecolor=edge[k],
                          linewidth=0.9),
            whiskerprops=dict(color=edge[k], linewidth=0.7),
            capprops=dict(color=edge[k], linewidth=0.7),
        )
        # legend handle
        axL.plot([], [], color=palette[k], marker="s", linestyle="none",
                 markersize=7, markeredgecolor=edge[k],
                 markeredgewidth=0.7, label=k)
        # overlay individual model points
        for pos, d in zip(positions, data):
            if not d:
                continue
            xs = np.full(len(d), pos) + np.random.uniform(-w * 0.18,
                                                          w * 0.18, len(d))
            axL.scatter(xs, d, s=5, color="black", alpha=0.45, zorder=3,
                        linewidths=0)
    axL.set_xticks(x)
    axL.set_xticklabels([q.replace("-", "-\n") for q in Q_TYPES])
    axL.set_ylabel("Hallucination rate (%)")
    axL.set_title("(b) Hallucination rate for each question type")
    axL.set_ylim(0, 105)
    axL.grid(axis="y", linestyle=":", alpha=0.4)
    axL.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.0,
               labelspacing=0.2)

    # ---- Right: per-grader lines vs conversation progress ----
    for k in keys:
        centers, rates, _ = agg_prog[k]
        axR.plot(centers * 100, rates * 100,
                 marker=markers[k], color=edge[k],
                 markerfacecolor=palette[k],
                 markersize=4.0, linewidth=1.6, label=k,
                 markeredgewidth=0.7)

    axR.set_xlabel("Conversation progress (%)")
    axR.set_ylabel("Hallucination rate (%)")
    axR.set_title("(c) Hallucination as conversation progresses")
    axR.set_xlim(0, 100)
    axR.grid(linestyle=":", alpha=0.4)
    axR.legend(frameon=False, loc="best", handlelength=1.6,
               labelspacing=0.25)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"[pdf] wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
