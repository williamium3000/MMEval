"""Replace Table 2 (Contextualization ablation) with a 1x3 bar plot.

Two ablation conditions, six evaluatee models, three metrics. Each panel
shows per-model delta vs. CEDI: 'no node selector' (lighter ablation)
in the blue family, 'noncontext' (no contextual scenario at all) in the
green family. Both ablations are negative on most cells; the plot makes
the relative magnitude obvious at a glance.
"""
import numpy as np
import matplotlib.pyplot as plt

OUT = "/raid/william/project/context-eval-mllm/nips_paper/fig/contextualization-ablation.pdf"

MODELS = ["LLaVA-7B", "InternVL2-8B", "InternVL2.5-8B", "InternVL3-8B",
          "Qwen2.5-7B", "gemma-3-12B"]

# (Numbers as in tab:exam-contextualization. None = "---" in the table.)
# Per-panel: (dict of {ablation: values}, y-axis unit label)
DATA = {
    "CHAIR$_I$ ($\\uparrow$)": (
        {
            "no node selector": [-2.8, -14.2, -13.5, -11.4, -6.0, -2.1],
            "noncontext":       [-23.0, -33.3, -36.0, -20.8, -21.3, -17.0],
        },
        r"$\Delta$ vs. CEDI (pp)",
    ),
    "Cov$_{\\mathrm{avg}}$ ($\\uparrow$)": (
        {
            "no node selector": [+2.8, -8.0, -9.3, -9.9, +7.0, +1.2],
            "noncontext":       [-15.6, -26.4, -25.3, -14.5, -5.7, -11.0],
        },
        r"$\Delta$ vs. CEDI (pp)",
    ),
    "GED ($\\uparrow$)": (
        {
            "no node selector": [-0.4, -11.1, -3.3, -9.1, -4.8, None],
            "noncontext":       [-17.3, -23.7, -16.3, -13.5, -10.7, -10.1],
        },
        r"$\Delta$ vs. CEDI (GED units)",
    ),
}

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

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))
    x = np.arange(len(MODELS))
    w = 0.38

    for ax, (metric, (schemes, ylabel)) in zip(axes, DATA.items()):
        v_node = [v if v is not None else np.nan for v in schemes["no node selector"]]
        v_nc   = [v if v is not None else np.nan for v in schemes["noncontext"]]
        ax.bar(x - w / 2, v_node, w, label="no node selector",
               facecolor=BLUE_FILL, edgecolor=BLUE_BORDER, linewidth=1.3)
        ax.bar(x + w / 2, v_nc, w, label="non-contextualized",
               facecolor=GREEN_FILL, edgecolor=GREEN_BORDER, linewidth=1.3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=30, ha="right",
                           rotation_mode="anchor")
        ax.set_title(metric)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", which="both", length=0)
        ax.grid(False)
        ax.margins(x=0.06)

    axes[-1].legend(frameon=False, loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[pdf] {OUT}")


if __name__ == "__main__":
    main()
