"""Per-turn hallucination plots derived from MMHal + VALOR.

Panel (a): distribution of hallucinated turns by question type
  (from VALOR unfaithful mentions on the no-history side).
Panel (b): mean per-turn hallucination rate vs. conversation
  progress, aggregating the two per-turn hallucination signals we
  have (MMHal has_hallucination and VALOR 1-faith_i) so a single
  metric's quirks don't dominate the curve.

Sample: the 3 evaluatees for which both per-turn signals are
available on v18/v18conv (InternVL2-8B, InternVL2.5-8B,
gemma-3-12B). Qwen2.5-VL-7B and LLaVA-1.5-7B are omitted from
this figure because their per-turn VALOR is only computed on the
SVG pipeline.

Two output PDFs:
  fig/error_analysis_valor.pdf  ---  panels (a) + (b) for the
    "hallucination as conversation progresses" subsection.
  fig/history_valor_boxplot.pdf ---  standalone box plot of
    VALOR 1-faith_i by question type, meant to sit next to
    tab:history-ablation.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from error_analysis import (
    Q_TYPES,
    REPO,
    per_model_rates_by_qtype,
    rate_by_qtype,
)


VALOR_PT_DIR    = os.path.join(REPO, "tmp/valor_pt")
OUT_MAIN_PDF    = os.path.join(REPO, "nips_paper/fig/error_analysis_valor.pdf")
OUT_BOX_PDF     = os.path.join(REPO, "nips_paper/fig/history_valor_boxplot.pdf")

N_BINS = 10


# ---------- collectors ------------------------------------------------

def _collect_valor(files, label):
    """files: {model_key: valor_pt_json_path}.  Returns rows tagged with
    signal = 1 - faithfulness_score_i per turn."""
    rows, used = [], []
    for m, p in files.items():
        if not os.path.isfile(p):
            print(f"  skip {m}: missing {p}")
            continue
        d = json.load(open(p))
        sents = d.get("sentences", [])
        if not sents:
            continue
        used.append(m)
        max_r = {}
        for s in sents:
            r = int(s.get("round_id") or 0)
            max_r[s.get("image_id")] = max(r, max_r.get(s.get("image_id"), 0))
        for s in sents:
            qt = s.get("q_type")
            if qt not in Q_TYPES:
                continue
            gen = s.get("generated_words") or []
            if not gen:
                continue
            faith = s.get("metrics", {}).get("faithfulness_score_i", 0.0)
            unfaith = max(0.0, min(1.0, 1.0 - float(faith)))
            r = int(s.get("round_id") or 0)
            mx = max_r.get(s.get("image_id"), 0) or 1
            rows.append((m, qt, r / mx, unfaith))
    print(f"[{label}] {len(used)} models, {len(rows)} per-turn records")
    return rows, used


def _collect_mmhal(models, base_dir, label):
    """models: {model_key: model_dir_name}, base_dir: the run root.
    Returns rows with signal = 1.0 if has_hallucination else 0.0.
    """
    rows, used = [], []
    for m, mdir in models.items():
        cands = [
            os.path.join(REPO, base_dir, mdir, f"mmhal_{mdir}.json"),
            os.path.join(REPO, base_dir, f"{mdir}_extracted",
                         f"mmhal_{mdir}_extracted.json"),
        ]
        p = next((c for c in cands if os.path.isfile(c)), None)
        if not p:
            print(f"  skip {m}: no mmhal file")
            continue
        d = json.load(open(p))
        results = d.get("detailed_results", [])
        if not results:
            continue
        used.append(m)
        from collections import defaultdict
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
            h = 1.0 if r.get("has_hallucination") else 0.0
            rows.append((m, qt, rid / mx, h))
    print(f"[{label}] {len(used)} models, {len(rows)} per-turn records")
    return rows, used


# Retain the old name for the box plot (renamed above)
_collect = _collect_valor


def rate_by_progress_per_model(rows, n_bins=N_BINS):
    """Mean per-turn rate per progress bin, equal-weighting each model."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    per_model = defaultdict(lambda: [[] for _ in range(n_bins)])
    for (m, _q, p, h) in rows:
        i = n_bins - 1 if p >= 1.0 else min(int(p * n_bins), n_bins - 1)
        per_model[m][i].append(h)
    bin_rates = np.full(n_bins, np.nan)
    for i in range(n_bins):
        model_means = [sum(bins[i]) / len(bins[i])
                        for bins in per_model.values() if bins[i]]
        if model_means:
            bin_rates[i] = sum(model_means) / len(model_means)
    return centers, bin_rates


# ---------- 5 evaluatees (same set as CHAIRi v2) ----------------------

NO_HIST = {
    "InternVL2-8B":              f"{VALOR_PT_DIR}/v18_InternVL2-8B.json",
    "InternVL2_5-8B":            f"{VALOR_PT_DIR}/v18_InternVL2_5-8B.json",
    "gemma-3-12b-it":            f"{VALOR_PT_DIR}/v18_gemma-3-12b-it.json",
    "Qwen2.5-VL-7B-Instruct":    f"{VALOR_PT_DIR}/svg_v19_Qwen2.5-VL-7B-Instruct.json",
    "llava-1.5-7b-hf":           f"{VALOR_PT_DIR}/svg_v19_llava-1.5-7b-hf.json",
}
WI_HIST = {
    "InternVL2-8B":              f"{VALOR_PT_DIR}/v18conv_InternVL2-8B.json",
    "InternVL2_5-8B":            f"{VALOR_PT_DIR}/v18conv_InternVL2_5-8B.json",
    "gemma-3-12b-it":            f"{VALOR_PT_DIR}/v18conv_gemma-3-12b-it.json",
    "Qwen2.5-VL-7B-Instruct":    f"{VALOR_PT_DIR}/svg_tt_v19conv_Qwen2.5-VL-7B-Instruct.json",
    "llava-1.5-7b-hf":           f"{VALOR_PT_DIR}/svg_tt_v19conv_llava-1.5-7b-hf.json",
}


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


PIE_COLORS = {"regular": "#D5E4F4", "follow-up": "#E4F0E0",
              "adversarial": "#F0D0C8", "unanswerable": "#F6E4B8"}
PIE_EDGE   = {"regular": "#2F6FCC", "follow-up": "#496F2C",
              "adversarial": "#B87474", "unanswerable": "#B08830"}
SERIES_COLOR = {"no history": "#D5E4F4", "with history": "#E4F0E0"}
SERIES_EDGE  = {"no history": "#2F6FCC", "with history": "#496F2C"}
SERIES_MARK  = {"no history": "o",       "with history": "D"}


def plot_main(no_valor, wi_valor, no_mmhal, wi_mmhal, out_pdf):
    """Two panels:
    (a) pie of hallucinated-turn share by q_type (VALOR unfaithful on
        no-history, since MMHal + VALOR share the same distribution
        story).
    (b) mean per-turn hallucination rate vs conversation progress,
        averaging the two per-turn signals (MMHal has_hallucination
        and VALOR 1-faith_i)."""
    fig, (axP, axR) = plt.subplots(
        1, 2, figsize=(7.5, 2.4),
        gridspec_kw={"width_ratios": [1, 2]},
    )

    counts = {qt: 0 for qt in Q_TYPES}
    for (_m, q, _p, h) in no_valor:
        if h > 0 and q in counts:
            counts[q] += 1
    slices, _, autotexts = axP.pie(
        [counts[qt] for qt in Q_TYPES],
        labels=[qt.replace("-", "-\n") for qt in Q_TYPES],
        colors=[PIE_COLORS[qt] for qt in Q_TYPES],
        wedgeprops=dict(edgecolor="white", linewidth=1.0),
        autopct="%1.0f%%", pctdistance=0.68,
        textprops=dict(fontsize=7),
    )
    for wedge, qt in zip(slices, Q_TYPES):
        wedge.set_edgecolor(PIE_EDGE[qt])
        wedge.set_linewidth(0.8)
    for t in autotexts:
        t.set_fontsize(6.5)
    axP.set_title("(a) Hallucination count by question type")

    for label, valor_rows, mmhal_rows in [
        ("no history",   no_valor, no_mmhal),
        ("with history", wi_valor, wi_mmhal),
    ]:
        centers, rate_v = rate_by_progress_per_model(valor_rows)
        _,       rate_m = rate_by_progress_per_model(mmhal_rows)
        combined = (rate_v + rate_m) / 2
        axR.plot(centers * 100, combined * 100,
                 marker=SERIES_MARK[label], color=SERIES_EDGE[label],
                 markerfacecolor=SERIES_COLOR[label],
                 markersize=4.0, linewidth=1.6, label=label,
                 markeredgewidth=0.7)
    axR.set_xlabel("Conversation progress (%)")
    axR.set_ylabel("Per-turn hallucination rate (%)")
    axR.set_title("(b) Hallucination as conversation progresses")
    axR.set_xlim(0, 100)
    axR.grid(linestyle=":", alpha=0.4)
    axR.legend(frameon=False, loc="best", handlelength=1.6,
               labelspacing=0.25)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[pdf] wrote {out_pdf}")
    plt.close(fig)


def plot_boxplot(no_rows, wi_rows, out_pdf):
    """Single panel: per-model box plot by q_type, two series.
    Taller-and-thinner than the standard 3-panel version so it fits
    next to a tall-thin table."""
    fig, axL = plt.subplots(figsize=(4.4, 3.0))
    box_data = {"no history":   per_model_rates_by_qtype(no_rows),
                "with history": per_model_rates_by_qtype(wi_rows)}
    x = np.arange(len(Q_TYPES))
    w = 0.34
    offsets = np.array([-w / 2, w / 2])
    for (label, off) in zip(["no history", "with history"], offsets):
        positions = x + off
        data = [[v * 100 for v in box_data[label][qt]] for qt in Q_TYPES]
        axL.boxplot(
            data, positions=positions, widths=w * 0.9,
            patch_artist=True, manage_ticks=False,
            showfliers=False, whis=1.5,
            medianprops=dict(color="black", linewidth=1.0),
            boxprops=dict(facecolor=SERIES_COLOR[label],
                          edgecolor=SERIES_EDGE[label], linewidth=0.9),
            whiskerprops=dict(color=SERIES_EDGE[label], linewidth=0.7),
            capprops=dict(color=SERIES_EDGE[label], linewidth=0.7),
        )
        axL.plot([], [], color=SERIES_COLOR[label], marker="s",
                 linestyle="none", markersize=7,
                 markeredgecolor=SERIES_EDGE[label],
                 markeredgewidth=0.7, label=label)
        for pos, d in zip(positions, data):
            if not d:
                continue
            xs = (np.full(len(d), pos)
                  + np.random.uniform(-w * 0.18, w * 0.18, len(d)))
            axL.scatter(xs, d, s=6, color="black", alpha=0.45, zorder=3,
                        linewidths=0)
    axL.set_xticks(x)
    axL.set_xticklabels([q.replace("-", "-\n") for q in Q_TYPES])
    axL.set_ylabel("Hallucination rate (%)")
    axL.set_ylim(0, 100)
    axL.grid(axis="y", linestyle=":", alpha=0.4)
    axL.legend(frameon=False, loc="upper left", ncol=1,
               handlelength=1.0, labelspacing=0.2)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[pdf] wrote {out_pdf}")
    plt.close(fig)


def main():
    # 5-model VALOR data (box plot uses full 5-model set on the paired
    # sources that already have per-turn VALOR outputs)
    no_valor_5, _ = _collect_valor(NO_HIST, "VALOR no-hist (5-model)")
    wi_valor_5, _ = _collect_valor(WI_HIST, "VALOR w-hist  (5-model)")

    # Combined-metric panel needs same source for both metrics per model.
    # MMHal is per-turn on v18/v18conv for all 5 models, but per-turn
    # VALOR on v18/v18conv only exists for the 3 non-SVG evaluatees.
    # Intersect.
    combo_models = {
        "InternVL2-8B":     "InternVL2-8B",
        "InternVL2_5-8B":   "InternVL2_5-8B",
        "gemma-3-12b-it":   "gemma-3-12b-it",
    }
    v18_root  = "work_dirs/vg/final_run_v18_gpt4o_completed"
    conv_root = "work_dirs/vg/final_run_v18_gpt4o_conv_completed"
    no_mmhal, _ = _collect_mmhal(combo_models, v18_root,  "MMHal no-hist")
    wi_mmhal, _ = _collect_mmhal(combo_models, conv_root, "MMHal w-hist")

    combo_valor_files_no = {m: NO_HIST[m] for m in combo_models}
    combo_valor_files_wi = {m: WI_HIST[m] for m in combo_models}
    no_valor_3, _ = _collect_valor(combo_valor_files_no, "VALOR no-hist (3-model)")
    wi_valor_3, _ = _collect_valor(combo_valor_files_wi, "VALOR w-hist  (3-model)")

    plot_main(no_valor_3, wi_valor_3, no_mmhal, wi_mmhal, OUT_MAIN_PDF)
    plot_boxplot(no_valor_5, wi_valor_5, OUT_BOX_PDF)

    for label, rows in [("VALOR no-hist", no_valor_3),
                        ("VALOR w-hist",  wi_valor_3),
                        ("MMHal no-hist", no_mmhal),
                        ("MMHal w-hist",  wi_mmhal)]:
        agg = rate_by_qtype(rows)
        s = " | ".join(f"{qt}={agg[qt][0]*100:5.1f}% (n={agg[qt][1]})"
                        for qt in Q_TYPES)
        print(f"  {label:16s}  {s}")


if __name__ == "__main__":
    main()
