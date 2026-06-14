"""Small-multiples slope chart: v19 vs v19ban2type ablation on VG.

One mini panel per grader-metric. Each panel has two x-positions
("v19", "ban2type") and three thin lines (one per evaluatee). Reading the
*mix* of slope directions across panels is the refutation: if removing the
two non-natural q-types (adversarial + unanswerable) drove all of CEDI's
hallucination exposure, every line in every panel would slope the same way.
Instead the grid shows "all-down / all-up / crossing" patterns side by side.

Data: tmp/ttcontext/_full_comparison.json (pulled from HF Icey444/ttcontext-result).
"""
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

DATA = "/tmp/ttcontext/_full_comparison.json"
OUT  = "/raid/william/project/context-eval-mllm/nips_paper/fig/ban2type_slope_grid.pdf"

MODELS = ["gemma-3-12b-it", "llava-1.5-7b-hf", "Qwen2.5-VL-7B-Instruct"]
SHORT  = {"gemma-3-12b-it": "gemma-3", "llava-1.5-7b-hf": "LLaVA",
          "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL"}

# Theme colors
BLUE_BORDER  = "#2F6FCC"
GREEN_BORDER = "#496F2C"
PINK_BORDER  = "#B87474"
MODEL_COLOR  = {"gemma-3-12b-it":            BLUE_BORDER,
                "llava-1.5-7b-hf":           GREEN_BORDER,
                "Qwen2.5-VL-7B-Instruct":    PINK_BORDER}

# (label, accessor, scale): scale multiplies stored value if it is a fraction.
PANELS = [
    ("agentic\\,(\\%)",         ("agentic_rate",),                           None),
    ("MMHal hall\\,(\\%)",      ("mmhal", "hall_rate"),                      100),
    ("CHAIR sent\\,(\\%)",      ("chair", "sentence_hall_rate"),             None),
    ("CHAIRi\\,(\\%)",          ("chair", "CHAIRi"),                         None),
    ("CHAIRi$_{v2}$\\,(\\%)",   ("chair", "CHAIRi_v2"),                      None),
    ("Cov$_{\\mathrm{avg}}$\\,(\\%)", ("chair", "Cov_avg"),                  None),
    ("VALOR obj faith",         ("valor_obj_exist", "faithfulness_score_i"), 100),
    ("VALOR obj cov",           ("valor_obj_exist", "coverage_score_i"),     100),
    ("VALOR rel faith",         ("valor_rel_pos", "faithfulness_score_i"),   100),
    ("VALOR rel cov",           ("valor_rel_pos", "coverage_score_i"),       100),
    ("SG $\\Delta$Con",         ("sg_delta_con", "avg"),                     None),
]


def get(node, keys):
    v = node
    for k in keys:
        v = v.get(k, {}) if isinstance(v, dict) else None
    if isinstance(v, dict):
        return None
    return v


def main():
    d = json.load(open(DATA))

    plt.rcParams.update({
        "font.family":     "serif",
        "font.size":       6.5,
        "axes.titlesize":  7.0,
        "axes.labelsize":  6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 5.5,
        "pdf.fonttype":    42,
        "ps.fonttype":     42,
    })

    nrow, ncol = 3, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5, 2.6),
                             constrained_layout=False)
    axes = axes.flatten()

    for idx, (label, key, scale) in enumerate(PANELS):
        ax = axes[idx]
        for m in MODELS:
            v19  = get(d[m]["v19"],         key)
            ban  = get(d[m]["v19ban2type"], key)
            if v19 is None or ban is None:
                continue
            if scale: v19, ban = v19 * scale, ban * scale
            ax.plot([0, 1], [v19, ban],
                    color=MODEL_COLOR[m], linewidth=1.0,
                    marker="o", markersize=2.5, markeredgewidth=0)

        ax.set_xlim(-0.25, 1.25)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["v19", "ban2"], fontsize=5.5)
        ax.tick_params(axis="x", which="both", length=0, pad=1)
        ax.tick_params(axis="y", which="both", length=2, pad=1)

        # Y-axis: only 2 ticks (min, max) drawn from the data extent for legibility
        ymin, ymax = ax.get_ylim()
        ax.set_yticks([ymin + 0.05*(ymax-ymin), ymax - 0.05*(ymax-ymin)])
        ax.set_yticklabels([f"{ymin + 0.05*(ymax-ymin):.0f}",
                            f"{ymax - 0.05*(ymax-ymin):.0f}"])

        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(0.5)
            ax.spines[s].set_color("#888")

        ax.set_title(label, pad=2)

    # Hide unused panel
    for k in range(len(PANELS), nrow * ncol):
        axes[k].axis("off")

    # Legend in the unused cell
    handles = [mpl.lines.Line2D([0], [0], color=MODEL_COLOR[m],
                                marker="o", markersize=3, linewidth=1.2,
                                label=SHORT[m])
               for m in MODELS]
    axes[-1].legend(handles=handles, loc="center", frameon=False,
                    fontsize=6.5, handlelength=1.2, labelspacing=0.4,
                    title="VG", title_fontsize=7)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.06,
                        hspace=0.55, wspace=0.45)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[pdf] {OUT}")


if __name__ == "__main__":
    main()
