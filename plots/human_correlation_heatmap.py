"""Per-image Pearson correlation between automatic graders and human-annotated
hallucination counts. Single-panel heatmap of the hallucination-side graders
(CHAIRi_v2, mmhal, n_hall_obj, GED_hal) using the paper theme fills:
pink (#F0D0C8) negative, white at 0, green (#E4F0E0) positive. GED itself
and the coverage-side graders are omitted from this view.

Rows: 4 evaluatees with human annotations.
Cols: CHAIRi_v2, mmhal, n_hall_obj (CHAIR raw count), GED_hal.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

OUT = "/raid/william/project/context-eval-mllm/nips_paper/fig/human_corr_heatmap.pdf"

# Theme colors (fills from plots/multiround_ablation.py): softer, match the rest
# of the paper's plot palette. Cell text reads cleanly on these lighter shades.
GREEN_FILL   = "#E4F0E0"
GREEN_BORDER = "#496F2C"
PINK_FILL    = "#F0D0C8"
PINK_BORDER  = "#B87474"
WHITE        = "#FFFFFF"
GRAY_NA      = "#EAEAEA"

MODELS = [
    "InternVL3-8B",
    "LLaVA-1.5-7B",
    "Opera-LLaVA-1.5",
    "Qwen3-VL-8B",
]

# Left panel: hallucination-side graders
HAL_GRADERS = [r"CHAIRi$_{v2}$", "mmhal", r"$n_{\mathrm{hall}}$",
               r"GED$_{\mathrm{hal}}$"]
HAL_RHO = np.array([
    # CHAIRi_v2   mmhal    n_hall_obj   GED_hal
    [-0.048,      0.268,   0.055,      0.322],   # InternVL3
    [ 0.045,     -0.024,   0.318,      0.328],   # LLaVA
    [-0.290,      np.nan, -0.029,      0.322],   # Opera (no mmhal)
    [-0.317,      0.332,  -0.241,      0.220],   # Qwen3-VL
])

def render_panel(ax, data, col_labels, row_labels, cmap, norm,
                 show_row_labels=True):
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    # Column ticks (top)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Row ticks
    ax.set_yticks(range(len(row_labels)))
    if show_row_labels:
        ax.set_yticklabels(row_labels)
    else:
        ax.set_yticklabels([""] * len(row_labels))
    ax.tick_params(axis="both", which="both", length=0)

    # Hide spines, draw thin white gridlines between cells
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)

    # Cell annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=6.5, color="#666")
                continue
            cell_color = cmap(norm(v))
            r, g, b = cell_color[:3]
            lum = 0.2126*r + 0.7152*g + 0.0722*b
            color = "#111" if lum > 0.6 else "white"
            txt = f"{v:+.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color=color)

    return im


def main():
    plt.rcParams.update({
        "font.family":     "serif",
        "font.size":       7.5,
        "axes.titlesize":  8,
        "axes.labelsize":  7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype":    42,
        "ps.fonttype":     42,
    })

    # Two-stop diverging cmap using the lighter theme fills.
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "pinkwhitegreen", [PINK_FILL, WHITE, GREEN_FILL]
    )
    cmap.set_bad(GRAY_NA)
    vmax = 0.40
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=+vmax)

    # Compact single-panel heatmap sized for a wrapfigure at ~0.5\linewidth.
    fig, ax = plt.subplots(1, 1, figsize=(2.3, 1.25))

    render_panel(ax, HAL_RHO, HAL_GRADERS, MODELS, cmap, norm,
                 show_row_labels=True)

    # Horizontal colorbar below
    cax = fig.add_axes([0.20, -0.18, 0.65, 0.06])
    cb = fig.colorbar(ax.images[0], cax=cax, orientation="horizontal",
                      ticks=[-vmax, 0, vmax])
    cb.set_label(r"per-image Pearson $\rho$ vs.\ human hall.\ count",
                 fontsize=6.5, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=2)

    fig.savefig(OUT, bbox_inches="tight")
    print(f"[pdf] {OUT}")


if __name__ == "__main__":
    main()
