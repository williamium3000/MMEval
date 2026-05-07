"""Compact bar plot for the Multi-round vs single-round ablation
(replaces tab:exam-multiround). Per-model delta vs CEDI on three
metrics; opera-1.5 row dropped (no data).
"""
import numpy as np
import matplotlib.pyplot as plt

OUT = "/raid/william/project/context-eval-mllm/nips_paper/fig/multiround-ablation.pdf"

MODELS = ["LLaVA-7B", "InternVL2-8B", "InternVL2.5-8B", "InternVL3-8B",
          "Qwen2.5-7B", "gemma-3-12B"]
DATA = {
    "CHAIRi fix":   [-15.4, -39.6, -35.9, -36.3, -39.7, -31.6],
    "Cov avg":      [-72.8, -74.3, -70.8, -73.8, -62.3, -75.3],
    "GED":          [-26.1, -34.8, -24.0, -27.5, -29.8, -26.2],
}

BLUE_FILL    = "#D5E4F4"
BLUE_BORDER  = "#2F6FCC"
GREEN_FILL   = "#E4F0E0"
GREEN_BORDER = "#496F2C"
PINK_FILL    = "#F0D0C8"
PINK_BORDER  = "#B87474"


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(1, 1, figsize=(3.0, 2.3))
    x = np.arange(len(MODELS))
    w = 0.27

    palette = [
        ("CHAIRi fix", BLUE_FILL,  BLUE_BORDER),
        ("Cov avg",    GREEN_FILL, GREEN_BORDER),
        ("GED",        PINK_FILL,  PINK_BORDER),
    ]
    offsets = np.linspace(-1, 1, 3) * w
    for (label, fill, edge), off in zip(palette, offsets):
        ax.bar(x + off, DATA[label], w,
               label=label, facecolor=fill, edgecolor=edge, linewidth=0.9)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=35, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel(r"$\Delta$ vs.\ CEDI (%)")
    ax.tick_params(axis="x", which="both", length=0)
    ax.grid(False)
    ax.margins(x=0.04)
    ax.legend(frameon=False, loc="lower right", handlelength=1.0,
              labelspacing=0.2, ncol=1)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[pdf] {OUT}")


if __name__ == "__main__":
    main()
