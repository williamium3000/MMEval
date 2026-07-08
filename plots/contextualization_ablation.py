"""Fig 5 (Contextualization ablation): 3-bar grouped plot.

For each of 6 evaluatees and 3 metrics, plot absolute values under
three conditions:
    * CEDI (full)             -- grey, the reference
    * w/o node selector       -- blue family
    * w/o context             -- green family

Deltas were previously plotted alone, which read as "condition A vs
condition B" rather than "each ablation vs the CEDI anchor". Plotting
the CEDI baseline as its own grey bar makes the comparison structure
obvious at a glance.

Absolute values reconstructed from the source rollup
(tmp/paper_results_index_rollup_with_sh.csv) plus the original
percentage-relative deltas that were hardcoded in the previous version
of this script; formula CEDI = abs_ablation / (1 + delta_pct/100).
Verified consistent to two decimals across both ablations.
"""
import numpy as np
import matplotlib.pyplot as plt

OUT = "/raid/william/project/context-eval-mllm/nips_paper/fig/contextualization-ablation.pdf"

MODELS = ["LLaVA-7B", "InternVL2-8B", "InternVL2.5-8B", "InternVL3-8B",
          "Qwen2.5-7B"]

METRICS = [
    ("CHAIR$_I$ ($\\uparrow$)",  "CHAIRi"),
    ("Cov. ($\\uparrow$)", "Cov_avg"),
    ("GED ($\\uparrow$)",        "GED"),
]

CEDI_ABS = {
    "LLaVA-7B":       {"CHAIRi": 51.20, "Cov_avg": 36.10, "GED":  88.87},
    "InternVL2-8B":   {"CHAIRi": 54.10, "Cov_avg": 44.30, "GED":  99.88},
    "InternVL2.5-8B": {"CHAIRi": 48.60, "Cov_avg": 41.30, "GED":  86.13},
    "InternVL3-8B":   {"CHAIRi": 48.20, "Cov_avg": 42.20, "GED":  90.38},
    "Qwen2.5-7B":     {"CHAIRi": 51.91, "Cov_avg": 41.40, "GED":  93.61},
    "gemma-3-12B":    {"CHAIRi": 52.20, "Cov_avg": 42.50, "GED":  91.51},
}
NONODE_ABS = {
    "LLaVA-7B":       {"CHAIRi": 49.77, "Cov_avg": 37.11, "GED":  88.51},
    "InternVL2-8B":   {"CHAIRi": 46.42, "Cov_avg": 40.76, "GED":  88.79},
    "InternVL2.5-8B": {"CHAIRi": 42.04, "Cov_avg": 37.46, "GED":  83.29},
    "InternVL3-8B":   {"CHAIRi": 42.71, "Cov_avg": 38.02, "GED":  82.16},
    "Qwen2.5-7B":     {"CHAIRi": 48.79, "Cov_avg": 44.30, "GED":  89.12},
    "gemma-3-12B":    {"CHAIRi": 51.10, "Cov_avg": 43.01, "GED":  None},
}
NC_ABS = {
    "LLaVA-7B":       {"CHAIRi": 39.42, "Cov_avg": 30.47, "GED":  73.50},
    "InternVL2-8B":   {"CHAIRi": 36.08, "Cov_avg": 32.60, "GED":  76.21},
    "InternVL2.5-8B": {"CHAIRi": 31.10, "Cov_avg": 30.85, "GED":  72.09},
    "InternVL3-8B":   {"CHAIRi": 38.17, "Cov_avg": 36.08, "GED":  78.18},
    "Qwen2.5-7B":     {"CHAIRi": 40.85, "Cov_avg": 39.04, "GED":  83.59},
    "gemma-3-12B":    {"CHAIRi": 43.33, "Cov_avg": 37.83, "GED":  82.27},
}

GREY_FILL    = "#DCDCDC"
GREY_BORDER  = "#7A7A7A"
BLUE_FILL    = "#D5E4F4"
BLUE_BORDER  = "#2F6FCC"
GREEN_FILL   = "#E4F0E0"
GREEN_BORDER = "#496F2C"


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    x = np.arange(len(MODELS))
    w = 0.27
    offsets = np.array([-1, 0, 1]) * w

    for ax, (title, key) in zip(axes, METRICS):
        cedi   = [CEDI_ABS[m][key]   if CEDI_ABS[m][key]   is not None else np.nan for m in MODELS]
        nonode = [NONODE_ABS[m][key] if NONODE_ABS[m][key] is not None else np.nan for m in MODELS]
        nc     = [NC_ABS[m][key]     if NC_ABS[m][key]     is not None else np.nan for m in MODELS]

        ax.bar(x + offsets[0], cedi,   w, label="CEDI (full)",
               facecolor=GREY_FILL,  edgecolor=GREY_BORDER,  linewidth=1.2)
        ax.bar(x + offsets[1], nonode, w, label="w/o node selector",
               facecolor=BLUE_FILL,  edgecolor=BLUE_BORDER,  linewidth=1.2)
        ax.bar(x + offsets[2], nc,     w, label="w/o context",
               facecolor=GREEN_FILL, edgecolor=GREEN_BORDER, linewidth=1.2)

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=30, ha="right",
                           rotation_mode="anchor")
        ax.set_title(title)
        ax.tick_params(axis="x", which="both", length=0)
        ax.grid(False)
        ax.margins(x=0.04, y=0.10)

    axes[0].set_ylabel(r"CHAIR$_I$ (%)")
    axes[1].set_ylabel(r"Cov. (%)")
    axes[2].set_ylabel(r"GED")

    axes[-1].legend(frameon=False, loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[pdf] {OUT}")


if __name__ == "__main__":
    main()
