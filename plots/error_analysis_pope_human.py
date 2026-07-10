"""Fig-11-style duplicates based on POPE and Human hallucination signals.

The main paper's Fig 11 (fig/error_analysis.pdf) is MMHal-flagged. This
script reuses the collectors from plots/error_analysis.py to build the
same 3-panel layout for two additional per-turn hallucination signals:

  * POPE  (per-question yes/no error rate, aggregated to per-turn).
          Only the v18 (no-history) evaluatee dirs have `_with_both_answers`
          files, so panel (b) and (c) show a single series and the caption
          notes that history-ablation isn't available for POPE.

  * Human (`len(turn["hallucination"]) > 0`).
          Two series: no-history (reference/human annotation/context/) and
          with-history (work_dirs/human/result/with_history/), when both
          variants exist for the same evaluatee.
"""
import os
import sys
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from error_analysis import (
    collect_pope,
    rate_by_qtype,
    per_model_rates_by_qtype,
    rate_by_progress,
    Q_TYPES,
    N_BINS,
    REPO,
)

# ---------- Human collectors (two dirs for no-hist / w-hist) ----------

HUMAN_NOHIST_DIR   = os.path.join(REPO, "reference/human annotation/context")
HUMAN_WITHHIST_DIR = os.path.join(REPO, "work_dirs/human/result/with_history")


def _collect_human_from(dir_path):
    rows, files = [], []
    if not os.path.isdir(dir_path):
        return rows, files
    for fn in sorted(f for f in os.listdir(dir_path) if f.endswith(".json")):
        model_key = (fn.replace("_single_first50.json", "")
                       .replace("_cache_first50.json", "")
                       .replace("_conversation(2).json", "")
                       .replace(".last6_first_conv.json", "")
                       .replace("_cache_conversation(2).json", "")
                       .replace(".json", ""))
        try:
            d = json.load(open(os.path.join(dir_path, fn)))
        except Exception as e:
            print(f"  skip {fn}: {e}")
            continue
        files.append(fn)
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
                rows.append((model_key, qt, rid / mx,
                             bool(t.get("hallucination"))))
    return rows, files


# ---------- Plot helper ----------

def _plot_3panel(series, out_pdf, source_label,
                 pie_series_key, palette, edge, markers,
                 title_suffix=""):
    """Render the Fig-11 3-panel layout for the supplied series dict.

    series: {name: list of (model, q_type, progress, hall_bool)}
    """
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

    keys = list(series.keys())
    fig, (axP, axL, axR) = plt.subplots(
        1, 3, figsize=(10.5, 2.4),
        gridspec_kw={"width_ratios": [1, 2, 2]},
    )

    # Panel (a): pie of hallucinated-turn counts by q_type on `pie_series_key`
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
    if sum(pie_counts) > 0:
        slices, _, autotexts = axP.pie(
            pie_counts,
            labels=[qt.replace("-", "-\n") for qt in Q_TYPES],
            colors=[pie_colors[qt] for qt in Q_TYPES],
            wedgeprops=dict(edgecolor="white", linewidth=1.0),
            autopct="%1.0f%%", pctdistance=0.68,
            textprops=dict(fontsize=7),
        )
        for wedge, qt in zip(slices, Q_TYPES):
            wedge.set_edgecolor(pie_edge[qt])
            wedge.set_linewidth(0.8)
        for t in autotexts:
            t.set_fontsize(6.5)
    else:
        axP.text(0.5, 0.5, "(no data)", ha="center", va="center",
                 transform=axP.transAxes)
    axP.set_title(f"(a) {source_label} hallucination count by q_type")

    # Panel (b): per-model boxplot by q_type
    box_data = {k: per_model_rates_by_qtype(series[k]) for k in keys}
    x = np.arange(len(Q_TYPES))
    w = 0.24
    offsets = (np.linspace(-(len(keys) - 1) / 2, (len(keys) - 1) / 2, len(keys))
               * w if len(keys) > 1 else np.array([0.0]))
    for k, off in zip(keys, offsets):
        positions = x + off
        data = [[v * 100 for v in box_data[k][qt]] for qt in Q_TYPES]
        axL.boxplot(
            data, positions=positions, widths=w * 0.9,
            patch_artist=True, manage_ticks=False,
            showfliers=False, whis=1.5,
            medianprops=dict(color="black", linewidth=1.0),
            boxprops=dict(facecolor=palette[k], edgecolor=edge[k],
                          linewidth=0.9),
            whiskerprops=dict(color=edge[k], linewidth=0.7),
            capprops=dict(color=edge[k], linewidth=0.7),
        )
        axL.plot([], [], color=palette[k], marker="s", linestyle="none",
                 markersize=7, markeredgecolor=edge[k],
                 markeredgewidth=0.7, label=k)
        for pos, d in zip(positions, data):
            if not d:
                continue
            xs = (np.full(len(d), pos)
                  + np.random.uniform(-w * 0.18, w * 0.18, len(d)))
            axL.scatter(xs, d, s=5, color="black", alpha=0.45, zorder=3,
                        linewidths=0)
    axL.set_xticks(x)
    axL.set_xticklabels([q.replace("-", "-\n") for q in Q_TYPES])
    axL.set_ylabel("Hallucination rate (%)")
    axL.set_title(f"(b) {source_label} rate by q_type")
    axL.set_ylim(0, 105)
    axL.grid(axis="y", linestyle=":", alpha=0.4)
    if len(keys) > 1:
        axL.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.0,
                   labelspacing=0.2)

    # Panel (c): rate vs progress
    agg_prog = {k: rate_by_progress(series[k]) for k in keys}
    for k in keys:
        centers, rates, _ = agg_prog[k]
        axR.plot(centers * 100, rates * 100,
                 marker=markers[k], color=edge[k],
                 markerfacecolor=palette[k],
                 markersize=4.0, linewidth=1.6, label=k,
                 markeredgewidth=0.7)
    axR.set_xlabel("Conversation progress (%)")
    axR.set_ylabel("Hallucination rate (%)")
    axR.set_title(f"(c) {source_label} rate vs. progress")
    axR.set_xlim(0, 100)
    axR.grid(linestyle=":", alpha=0.4)
    if len(keys) > 1:
        axR.legend(frameon=False, loc="best", handlelength=1.6,
                   labelspacing=0.25)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[pdf] wrote {out_pdf}")
    plt.close(fig)


def main():
    # ---- POPE (single series — no v18conv POPE files exist) ----
    pope_rows = collect_pope()
    series_pope = {"POPE (v18)": pope_rows}
    _plot_3panel(
        series_pope,
        out_pdf=os.path.join(REPO, "nips_paper/fig/error_analysis_pope.pdf"),
        source_label="POPE",
        pie_series_key="POPE (v18)",
        palette={"POPE (v18)": "#D5E4F4"},
        edge={"POPE (v18)": "#2F6FCC"},
        markers={"POPE (v18)": "o"},
    )
    # print summary
    for qt in Q_TYPES:
        rate, n = rate_by_qtype(pope_rows)[qt]
        print(f"  POPE {qt}: {rate*100:.1f}% (n={n})")

    # ---- Human (up to 2 series depending on directory availability) ----
    nohist_rows, nohist_files = _collect_human_from(HUMAN_NOHIST_DIR)
    withhist_rows, withhist_files = _collect_human_from(HUMAN_WITHHIST_DIR)
    print(f"[human no-hist]  {len(nohist_files)} files, {len(nohist_rows)} rows")
    print(f"[human w/-hist]  {len(withhist_files)} files, {len(withhist_rows)} rows")

    series_human = {}
    if nohist_rows:
        series_human["Human -- no history"] = nohist_rows
    if withhist_rows:
        series_human["Human -- with history"] = withhist_rows

    if not series_human:
        print("no human data available, skipping human plot")
        return

    _plot_3panel(
        series_human,
        out_pdf=os.path.join(REPO, "nips_paper/fig/error_analysis_human.pdf"),
        source_label="Human",
        pie_series_key=("Human -- no history" if "Human -- no history"
                        in series_human else list(series_human.keys())[0]),
        palette={
            "Human -- no history":   "#D5E4F4",
            "Human -- with history": "#E4F0E0",
        },
        edge={
            "Human -- no history":   "#2F6FCC",
            "Human -- with history": "#496F2C",
        },
        markers={
            "Human -- no history":   "o",
            "Human -- with history": "D",
        },
    )
    for k in series_human:
        for qt in Q_TYPES:
            rate, n = rate_by_qtype(series_human[k])[qt]
            print(f"  {k} {qt}: {rate*100:.1f}% (n={n})")


if __name__ == "__main__":
    main()
