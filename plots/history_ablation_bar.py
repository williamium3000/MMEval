"""Relative bar plot for the history ablation (tab:history-ablation -> figure).

Each metric's w/-hist mean is shown relative to its w/o-hist mean (= 1.0),
so metrics with very different scales (%, GED, DeltaCon) share one axis.
Values are the per-metric means over 5 evaluatees hardcoded from the paper
table (see result.tex history-ablation paragraph for provenance).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PDF = os.path.join(REPO, "nips_paper/fig/history_ablation_bar.pdf")

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

BAR_FACE = "#E4F0E0"
BAR_EDGE = "#496F2C"
BASE_COLOR = "#2F6FCC"

# metric label, w/o hist, w/ hist
METRICS = [
    (r"CHAIR$_I$ ($\uparrow$)",         49.0, 31.2),
    (r"MMHal ($\uparrow$)",             38.4, 50.1),
    (r"VALOR faith$_i$ ($\downarrow$)", 68.6, 63.0),
    (r"GED ($\uparrow$)",               91.9, 123.6),
    (r"SG $\Delta$Con ($\uparrow$)",    4.36, 4.73),
]


def main():
    labels = [m[0] for m in METRICS]
    ratios = [wi / wo for (_l, wo, wi) in METRICS]

    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    y = np.arange(len(METRICS))[::-1]
    ax.barh(y, ratios, height=0.62, color=BAR_FACE,
            edgecolor=BAR_EDGE, linewidth=0.9, zorder=3)
    ax.axvline(1.0, color=BASE_COLOR, linewidth=1.0,
               linestyle="--", zorder=2)

    def fmt(v):
        return f"{v:.2f}" if v < 10 else f"{v:.1f}"

    for yi, r, (_l, wo, wi) in zip(y, ratios, METRICS):
        ax.text(r + 0.03, yi, f"{fmt(wo)} $\\rightarrow$ {fmt(wi)}",
                va="center", ha="left", fontsize=7, color="black")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("w/ hist relative to w/o hist (= 1.0)")
    ax.set_xlim(0, 1.75)
    ax.text(1.0, len(METRICS) - 0.32, "w/o hist", color=BASE_COLOR,
            fontsize=7, ha="center", va="bottom")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="gray", alpha=0.18, linewidth=0.5, zorder=0)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PDF.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print("wrote", OUT_PDF)
    for (l, wo, wi), r in zip(METRICS, ratios):
        print(f"{l:35s} {wo:8.2f} -> {wi:8.2f}  ratio {r:.3f}")


if __name__ == "__main__":
    main()
