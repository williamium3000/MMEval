"""Fig 7 (Human realism rankings): 3 stacked-bar conditions (CEDI, Caption,
POPE), each stack showing Rank 1 / 2 / 3 shares. Restyled to match the paper's
plot palette (light fill + darker mid + dark border) and serif typography
shared with multiround_ablation.py, contextualization_ablation.py, etc.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = "/raid/william/project/context-eval-mllm/nips_paper/fig/realism_ranking_figure.pdf"

# Paper palette (matches other plot scripts in plots/):
#   fill (lightest)  -> mid (rank 2) -> border (darkest, rank 1)
PALETTE = {
    "CEDI":    ("#D5E4F4", "#6F95D8", "#2F6FCC"),   # blue family
    "Caption": ("#F0D0C8", "#D19E96", "#B87474"),   # pink family
    "POPE":    ("#E4F0E0", "#8FB27D", "#496F2C"),   # green family
}

# Values transcribed from the previous realism_ranking_figure.pdf.
DATA = {
    #             Rank1(best), Rank2,  Rank3(worst)
    "CEDI":     [46, 30, 24],
    "Caption":  [32, 36, 32],
    "POPE":     [22, 34, 44],
}
CONDITIONS = ["CEDI", "Caption", "POPE"]
RANK_LABELS = ["Rank 1 (Best)", "Rank 2", "Rank 3 (Worst)"]

# CEDI vs POPE significance marker
SIG_TEXT  = r"$^{*}p = .011$"
SIG_PAIR  = (0, 2)   # indices into CONDITIONS

def main():
    plt.rcParams.update({
        "font.family":     "serif",
        "font.size":       10,
        "axes.titlesize":  11,
        "axes.labelsize":  10,
        "xtick.labelsize": 10,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.3))
    x = np.arange(len(CONDITIONS))
    w = 0.55

    for i, cond in enumerate(CONDITIONS):
        fill, mid, border = PALETTE[cond]
        r1, r2, r3 = DATA[cond]
        # Stack: Rank 1 (bottom, darkest), Rank 2 (mid), Rank 3 (top, lightest).
        ax.bar(x[i], r1, w, facecolor=border, edgecolor=border, linewidth=1.1)
        ax.bar(x[i], r2, w, bottom=r1,
               facecolor=mid, edgecolor=border, linewidth=1.1)
        ax.bar(x[i], r3, w, bottom=r1 + r2,
               facecolor=fill, edgecolor=border, linewidth=1.1)

        # In-bar labels, colour chosen for legibility on each segment.
        ax.text(x[i], r1 / 2,               f"{r1}%", ha="center", va="center",
                color="white", fontsize=10)
        ax.text(x[i], r1 + r2 / 2,          f"{r2}%", ha="center", va="center",
                color="#111", fontsize=10)
        ax.text(x[i], r1 + r2 + r3 / 2,     f"{r3}%", ha="center", va="center",
                color="#111", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel("Percentage of Annotations (%)")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="x", which="both", length=0)

    # Significance bracket: CEDI (i=0) --- POPE (i=2), above the bars.
    i, j = SIG_PAIR
    y_bracket = 106
    ax.plot([x[i], x[i], x[j], x[j]],
            [y_bracket - 3, y_bracket, y_bracket, y_bracket - 3],
            color="#333", linewidth=0.8)
    ax.text((x[i] + x[j]) / 2, y_bracket + 2, SIG_TEXT,
            ha="center", va="bottom", fontsize=10)

    # Rank legend (grey chips to keep the legend condition-agnostic).
    from matplotlib.patches import Patch
    grey_border = "#555555"
    handles = [
        Patch(facecolor=grey_border, edgecolor=grey_border, label=RANK_LABELS[0]),
        Patch(facecolor="#A6A6A6",   edgecolor=grey_border, label=RANK_LABELS[1]),
        Patch(facecolor="#DCDCDC",   edgecolor=grey_border, label=RANK_LABELS[2]),
    ]
    ax.legend(handles=handles, frameon=False, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, -0.28),
              handlelength=1.4, columnspacing=1.4, borderaxespad=0.0)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[pdf] {OUT}")


if __name__ == "__main__":
    main()
