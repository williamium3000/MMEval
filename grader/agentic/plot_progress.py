"""Aggregate agentic hallucination predictions by conversation progress and
plot the (b)-style "hallucination rate vs progress %" figure for v19 (no
history) vs v19conv (with history), averaged across the 5 svg overlap models.

Usage:
    python -m grader.agentic.plot_progress \
        --no-history-dir grader/agentic/predictions/v19_n50 \
        --with-history-dir grader/agentic/predictions/v19conv_n50 \
        --output grader/agentic/predictions/plot_progress.png

Each prediction JSON is a list of image entries; each entry has
``conversations: [{round_id, prompt, response, q_type, hallucination, ...}]``
where ``hallucination`` is a list of {hallucination, reason}. A round is
"hallucinated" iff that list is non-empty.

Progress for round i out of N rounds in a conversation is computed as the
bin center ``(i + 0.5) / N`` (i 0-indexed), then assigned to one of 10 bins
of width 10% (centers at 5, 15, ..., 95%).
"""
import argparse
import glob
import json
import os
from collections import defaultdict


BIN_EDGES = [i * 10 for i in range(11)]   # 0,10,...,100
BIN_CENTERS = [b + 5 for b in BIN_EDGES[:-1]]  # 5,15,...,95


def _bin_index(progress_pct):
    """progress_pct in [0,100]; returns index in 0..9."""
    if progress_pct >= 100:
        return 9
    if progress_pct < 0:
        return 0
    return int(progress_pct // 10)


def aggregate(pred_dir):
    """Return (rates, counts) lists of length 10, plus per-model dict."""
    bin_hall = [0] * 10
    bin_tot = [0] * 10
    per_model = {}
    files = sorted(glob.glob(os.path.join(pred_dir, "*.pred.json")))
    if not files:
        raise SystemExit(f"no *.pred.json under {pred_dir}")
    for f in files:
        model = os.path.basename(f).replace(".pred.json", "")
        m_hall = [0] * 10
        m_tot = [0] * 10
        data = json.load(open(f))
        for img in data:
            convs = img.get("conversations", [])
            n = len(convs)
            if n == 0:
                continue
            for i, c in enumerate(convs):
                progress = (i + 0.5) / n * 100.0
                b = _bin_index(progress)
                hallucinated = bool(c.get("hallucination") or [])
                m_tot[b] += 1
                bin_tot[b] += 1
                if hallucinated:
                    m_hall[b] += 1
                    bin_hall[b] += 1
        per_model[model] = {
            "rate": [(h / t * 100 if t else 0.0) for h, t in zip(m_hall, m_tot)],
            "count": list(m_tot),
        }
    rates = [(h / t * 100 if t else 0.0) for h, t in zip(bin_hall, bin_tot)]
    return rates, bin_tot, per_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-history-dir", required=True)
    ap.add_argument("--with-history-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--title", default="(b) By conversation progress")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    no_rates, no_counts, no_pm = aggregate(args.no_history_dir)
    wh_rates, wh_counts, wh_pm = aggregate(args.with_history_dir)

    print(f"no-history bin totals : {no_counts}")
    print(f"with-history bin totals: {wh_counts}")
    print(f"no-history overall rate : "
          f"{sum(no_counts) and 100*sum(no_pm[k]['rate'][b]*no_pm[k]['count'][b] for k in no_pm for b in range(10)) /  (100*sum(no_counts)):.1f}%")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(BIN_CENTERS, no_rates, "o-", color="#3478b8",
            label="agentic -- no history",  linewidth=2, markersize=7)
    ax.plot(BIN_CENTERS, wh_rates, "D-", color="#5b8a3a",
            label="agentic -- with history", linewidth=2, markersize=7)
    ax.set_xlabel("Conversation progress (%)")
    ax.set_ylabel("Hallucination rate (%)")
    ax.set_title(args.title)
    ax.set_xlim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")

    # also dump JSON of the binned values for the record
    out_json = os.path.splitext(args.output)[0] + ".json"
    json.dump({
        "bin_centers": BIN_CENTERS,
        "no_history":   {"rate": no_rates,  "count": no_counts, "per_model": no_pm},
        "with_history": {"rate": wh_rates, "count": wh_counts, "per_model": wh_pm},
    }, open(out_json, "w"), indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
