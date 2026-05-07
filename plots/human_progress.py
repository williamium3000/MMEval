"""Single-panel companion to fig:error-analysis: human-annotation
hallucination-spans-per-turn vs. conversation progress, comparing the
no-history CEDI run against the with-history CEDI run on the same model."""
import json
import numpy as np
import matplotlib.pyplot as plt

OUT = "/raid/william/project/context-eval-mllm/nips_paper/fig/error_analysis_human_progress.pdf"

R_FILES = [
    ("human -- no history",
     "/raid/william/project/context-eval-mllm/work_dirs/human/vg/"
     "final_run_v18_gpt4o_completed/opera-llava-1.5_cache_first50.json",
     "#2F6FCC"),
    ("human -- CEDI with history",
     "/raid/william/project/context-eval-mllm/work_dirs/human/vg/"
     "final_run_v18_gpt4o_conv_completed/opera-llava-1.5_cache_conversation(2).json",
     "#496F2C"),
]
N_BINS = 10


def progress_curve(path, n_bins=N_BINS):
    d = json.load(open(path))
    rows = []
    for s in d:
        convs = s.get("conversations") or []
        if not convs:
            continue
        mx = max((t.get("round_id") or 0) for t in convs) or 1
        for t in convs:
            rid = int(t.get("round_id") or 0)
            rows.append((rid / mx, len(t.get("hallucination") or [])))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    rates = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sub = [v for (p, v) in rows
               if (lo <= p <= hi if i == n_bins - 1 else lo <= p < hi)]
        rates.append(np.mean(sub) if sub else np.nan)
    return centers, np.array(rates)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 2.6))
    for label, path, color in R_FILES:
        c, r = progress_curve(path)
        ax.plot(c * 100, r, marker="o", color=color,
                markersize=4.0, linewidth=1.6, label=label,
                markeredgecolor="black", markeredgewidth=0.4)
    ax.set_xlabel("Conversation progress (%)")
    ax.set_ylabel("Hallucinations per turn")
    ax.set_xlim(0, 100)
    ax.grid(False)
    ax.legend(frameon=False, loc="best", handlelength=1.6, labelspacing=0.25)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[pdf] {OUT}")


if __name__ == "__main__":
    main()
