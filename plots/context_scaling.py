"""Re-plot Figure 4: effect of the number of grounded contexts per image
(2 vs 5) on CHAIRi-fix, Cov-avg, GED across five evaluatee models.

Uses the user-supplied theme: light-blue fill / dark-blue border for the
2-context bars, light-green fill / dark-green border for the 5-context bars.
No grid, larger labels, shorter aspect ratio.
"""
import numpy as np
import matplotlib.pyplot as plt

OUT = "/raid/william/project/context-eval-mllm/nips_paper/fig/context-scaling.pdf"

MODELS = [
    "LLaVA-1.5-7B", "InternVL2-8B", "InternVL2.5-8B", "InternVL3-8B",
    "Qwen2.5-VL-7B",
]
# Values transcribed from the previous figure (per metric, per model).
DATA = {
    "CHAIR$_I$ ($\\uparrow$)": {
        "2": [0.512, 0.541, 0.488, 0.482, 0.520],
        "5": [0.578, 0.602, 0.555, 0.543, 0.612],
    },
    "Cov. ($\\uparrow$)": {
        "2": [0.362, 0.443, 0.413, 0.422, 0.413],
        "5": [0.542, 0.610, 0.583, 0.578, 0.626],
    },
    "GED ($\\uparrow$)": {
        "2": [89, 100, 86, 90, 94],
        "5": [142, 155, 130, 131, 143],
    },
}

# Theme
BLUE_FILL    = "#D5E4F4"
BLUE_BORDER  = "#2F6FCC"
BLUE_HATCH   = "#6F95D8"
GREEN_FILL   = "#E4F0E0"
GREEN_BORDER = "#496F2C"
GREEN_HATCH  = "#6EA143"


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

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))
    x = np.arange(len(MODELS))
    w = 0.38

    # Compact model labels for x-axis
    short = {
        "LLaVA-1.5-7B":   "LLaVA-1.5",
        "InternVL2-8B":   "InternVL2",
        "InternVL2.5-8B": "InternVL2.5",
        "InternVL3-8B":   "InternVL3",
        "Qwen2.5-VL-7B":  "Qwen2.5-VL",
    }

    for ax, (metric, series) in zip(axes, DATA.items()):
        v2 = series["2"]
        v5 = series["5"]
        ax.bar(x - w / 2, v2, w, label="2 contexts",
               facecolor=BLUE_FILL, edgecolor=BLUE_BORDER, linewidth=1.3)
        ax.bar(x + w / 2, v5, w, label="5 contexts",
               facecolor=GREEN_FILL, edgecolor=GREEN_BORDER, linewidth=1.3)
        ax.set_xticks(x)
        ax.set_xticklabels([short[m] for m in MODELS], rotation=30,
                           ha="right", rotation_mode="anchor")
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", which="both", length=0)
        ax.grid(False)
        ax.margins(x=0.06)

    # Place legend outside, to the right of the rightmost panel.
    axes[-1].legend(frameon=False, loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0.0)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[pdf] {OUT}")


if __name__ == "__main__":
    main()
