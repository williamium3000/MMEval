"""
Paired comparison: long-context examiner (full conversation history fed back to
the examinee) vs. the equivalent text-only v18 examiner, on the 10 models
that were run under both settings.

Per-turn hallucination signal = MMHal `has_hallucination` (the only grader
available for both settings). Reports overall and per-q_type rates per model.

Outputs:
  plots/long_ctx_paired.csv   -- machine-readable
  plots/long_ctx_paired.tex   -- LaTeX-ready table chunk
"""
import json
import os
from collections import defaultdict

REPO = "/raid/william/project/context-eval-mllm"
V18_DIR  = os.path.join(REPO, "work_dirs/vg/final_run_v18_gpt4o_completed")
LONG_DIR = os.path.join(REPO, "work_dirs/vg/final_run_v18_gpt4o_conv_completed")

Q_TYPES = ["regular", "follow-up", "adversarial", "unanswerable"]


def load_mmhal(d, model_subdir):
    p = os.path.join(d, model_subdir, f"mmhal_{model_subdir}.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f).get("detailed_results", [])


def rate(rows, qt=None):
    sub = [r for r in rows if (qt is None or r.get("q_type") == qt)]
    if not sub:
        return None, 0
    n_hal = sum(1 for r in sub if r.get("has_hallucination"))
    return n_hal / len(sub), len(sub)


def main():
    # Models with mmhal jsons in the long-ctx run
    long_models = sorted(
        m for m in os.listdir(LONG_DIR)
        if os.path.isdir(os.path.join(LONG_DIR, m))
        and os.path.isfile(os.path.join(LONG_DIR, m, f"mmhal_{m}.json"))
    )
    print(f"long-ctx models: {len(long_models)}")

    rows = []
    for m in long_models:
        v18  = load_mmhal(V18_DIR,  m)
        long = load_mmhal(LONG_DIR, m)
        if v18 is None:
            print(f"  skip {m}: no v18 mmhal")
            continue
        rec = {"model": m}
        for label, src in [("v18", v18), ("long", long)]:
            r_all, n_all = rate(src)
            rec[f"{label}_all"] = (r_all, n_all)
            for qt in Q_TYPES:
                rec[f"{label}_{qt}"] = rate(src, qt)
        rows.append(rec)

    # ---- CSV ----
    out_csv = os.path.join(REPO, "plots/long_ctx_paired.csv")
    with open(out_csv, "w") as f:
        cols = ["model", "v18_all", "long_all", "delta_all"]
        for qt in Q_TYPES:
            cols += [f"v18_{qt}", f"long_{qt}", f"delta_{qt}"]
        f.write(",".join(cols) + "\n")
        for rec in rows:
            v_all, _ = rec["v18_all"]
            l_all, _ = rec["long_all"]
            line = [rec["model"], f"{v_all:.4f}", f"{l_all:.4f}",
                    f"{l_all - v_all:+.4f}"]
            for qt in Q_TYPES:
                v, _ = rec[f"v18_{qt}"]
                l, _ = rec[f"long_{qt}"]
                line += [f"{v:.4f}" if v is not None else "",
                         f"{l:.4f}" if l is not None else "",
                         f"{l - v:+.4f}" if (v is not None and l is not None) else ""]
            f.write(",".join(line) + "\n")
    print(f"[csv] {out_csv}")

    # ---- Print summary ----
    print(f"\n{'model':32s} {'all':>14s} {'reg':>14s} {'foll':>14s} {'adv':>14s} {'una':>14s}")
    deltas = defaultdict(list)
    for rec in rows:
        cells = []
        for qt_label, key in [("all","_all"),
                              ("reg","_regular"),
                              ("foll","_follow-up"),
                              ("adv","_adversarial"),
                              ("una","_unanswerable")]:
            v, _ = rec["v18" + key]
            l, _ = rec["long" + key]
            if v is None or l is None:
                cells.append("           --")
                continue
            d = l - v
            deltas[qt_label].append(d)
            cells.append(f"{v*100:5.1f}->{l*100:5.1f}")
        print(f"{rec['model']:32s} " + " ".join(f"{c:>14s}" for c in cells))

    # mean delta
    print("\nmean delta:")
    for qt_label in ("all","reg","foll","adv","una"):
        ds = deltas[qt_label]
        if ds:
            mu = sum(ds) / len(ds) * 100
            print(f"  {qt_label:>4s}: {mu:+.2f} pp (n_models={len(ds)})")

    # ---- LaTeX ----
    out_tex = os.path.join(REPO, "plots/long_ctx_paired.tex")
    pretty = {
        "InternVL2-2B": "InternVL2-2B",
        "InternVL2-8B": "InternVL2-8B",
        "InternVL2_5-2B": "InternVL2.5-2B",
        "InternVL2_5-8B": "InternVL2.5-8B",
        "InternVL3-2B-Instruct": "InternVL3-2B",
        "Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B",
        "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
        "gemma-3-4b-it": "gemma-3-4B",
        "gemma-3-12b-it": "gemma-3-12B",
        "llava-1.5-7b-hf": "LLaVA-1.5-7B",
    }
    with open(out_tex, "w") as f:
        f.write(r"""\begin{table}[h]
\centering
\tiny
\setlength{\tabcolsep}{4pt}
\caption{Paired long-context vs.\ text-only examiner on the 10 models that were
run under both settings. Cells show the per-turn MMHal hallucination rate
(\%); $\Delta$ is the long-context rate minus the v18 rate. Positive
$\Delta$ means long context elicited \emph{more} hallucinations.}
\label{tab:exam-longctx-paired}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{ALL} & \textbf{regular} & \textbf{follow-up} & \textbf{adversarial} & \textbf{unanswerable} \\
\midrule
""")
        for rec in rows:
            name = pretty.get(rec["model"], rec["model"])
            cells = [name]
            for key in ("_all", "_regular", "_follow-up", "_adversarial", "_unanswerable"):
                v, _ = rec["v18" + key]
                l, _ = rec["long" + key]
                if v is None or l is None:
                    cells.append("---")
                    continue
                d = (l - v) * 100
                arrow = (r"\textcolor{red!70!black}{$\triangle$ +%.1f}" % d
                         if d > 0
                         else r"\textcolor{green!60!black}{$\triangledown$ %.1f}" % d)
                cells.append(f"{v*100:.1f} $\\to$ {l*100:.1f}~~{arrow}")
            f.write(" & ".join(cells) + r" \\" + "\n")
        # mean row
        f.write(r"\midrule" + "\n")
        cells = [r"\textbf{mean $\Delta$}"]
        for qt_label in ("all","regular","follow-up","adversarial","unanswerable"):
            qt_short = {"all":"all","regular":"reg","follow-up":"foll",
                        "adversarial":"adv","unanswerable":"una"}[qt_label]
            ds = deltas[qt_short]
            if not ds:
                cells.append("---"); continue
            mu = sum(ds) / len(ds) * 100
            arrow = (r"\textcolor{red!70!black}{$\triangle$ +%.1f}" % mu
                     if mu > 0
                     else r"\textcolor{green!60!black}{$\triangledown$ %.1f}" % mu)
            cells.append(arrow)
        f.write(" & ".join(cells) + r" \\" + "\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print(f"[tex] {out_tex}")


if __name__ == "__main__":
    main()
