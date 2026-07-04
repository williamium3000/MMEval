"""
Per-image Pearson correlation between human-annotated hallucination rate and
each of the 9 automatic graders (CHAIRs, CHAIRi, 1-Cov, MMHal, GED, DELCON,
SoftSPICE, HaELM, Faith), split by examiner (baseline vs CEDI).

Models with human annotation:
  baseline (3 models): LLaVA-1.5-7B, InternVL3-8B-Instruct, Opera-LLaVA-1.5
  CEDI     (4 models): + Qwen3-VL-8B-Instruct

Per-image features:
  human   : fraction of turns with >=1 annotated hallucination span
  CHAIRs  : binary per-image, hallucinated_words `metrics.CHAIRs`
  CHAIRi  : per-image, hallucinated_words `metrics.CHAIRi`
  1-Cov   : 1 - per-image coverage
  MMHal   : per-record_index halluc rate (record matched to image by 1st prompt)
  GED, DELCON       : per-sample `dist_score` from sg_*.json
  SoftSPICE         : per-sample fraction from softspice/spice_<model>.json
  HaELM             : fraction of turns with `accurate` extracted as 'no' (across that record)
  Faith             : per-sample faithfulness from faithscore/<model>_*.json

Outputs:
  plots/human_grader_correlation.csv
  plots/human_grader_correlation.tex
"""
import json
import os
from collections import defaultdict
from math import sqrt

REPO = "/raid/william/project/context-eval-mllm"
HUMAN_BASELINE = os.path.join(REPO, "reference/human annotation/baseline")
HUMAN_CEDI     = os.path.join(REPO, "reference/human annotation/context")

BASELINE_MODELS = [
    ("InternVL3-8B-Instruct_baseline_processed_first50.json", "InternVL3-8B-Instruct"),
    ("llava-1.5-7b-hf_baseline_processed_first50.json",       "llava-1.5-7b-hf"),
    ("opera-llava-1.5_done.json",                              "opera-llava-1.5"),
]
CEDI_MODELS = [
    ("InternVL3-8B-Instruct_single_first50.json",  "InternVL3-8B-Instruct"),
    ("Qwen3-VL-8B-Instruct_single_first50.json",   "Qwen3-VL-8B-Instruct"),
    ("llava-1.5-7b-hf_single_first50.json",        "llava-1.5-7b-hf"),
    ("opera-llava-1.5_cache_first50.json",         "opera-llava-1.5"),
]

BASELINE_GRADER_DIR = os.path.join(REPO, "work_dirs/vg/ablation_baseline2_completed")
CEDI_GRADER_DIR     = os.path.join(REPO, "work_dirs/vg/final_run_v18_gpt4o_completed")

# Some graders for opera live in a doubly-nested subdir (model_subdir/model_subdir/<files>)
NESTED_OPERA = "opera-llava-1.5"
MMHAL_OVERRIDE = {  # CEDI dir places opera mmhal under uppercase folder
    ("CEDI", "opera-llava-1.5"): ("Opera-LLaVA-1.5", "mmhal_Opera-LLaVA-1.5.json"),
}


# ---------------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------------

def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None, len(pairs)
    n = len(pairs)
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    dy = sqrt(sum((y - my) ** 2 for _, y in pairs))
    if dx == 0 or dy == 0:
        return None, n
    return num / (dx * dy), n


# ---------------------------------------------------------------------------
# File-resolution helpers (some files live in nested subdirs for opera)
# ---------------------------------------------------------------------------

def find_grader_file(grader_dir, model_subdir, fname_template):
    """Look for <fname> in standard <grader_dir>/<model> location, then in
    nested <grader_dir>/<model>/<model>/ (used for opera in some sweeps),
    and finally in <grader_dir>/<model>_merged_internal_sg/ or <model>_sg/
    (used by the recent SG-distance sweep)."""
    f = fname_template.format(model=model_subdir)
    candidates = [
        os.path.join(grader_dir, model_subdir, f),
        os.path.join(grader_dir, model_subdir, model_subdir, f),
    ]
    for suffix in ("_merged_internal_sg", "_sg"):
        sub = model_subdir + suffix
        f2 = fname_template.format(model=sub)
        candidates.append(os.path.join(grader_dir, sub, f2))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Loaders -> per-image dicts {image_id: value}
# ---------------------------------------------------------------------------

def human_per_image(path):
    """Per-image human signal = mean number of annotated hallucination spans
    per turn (count-normalized, not binary)."""
    out = {}
    d = json.load(open(path))
    for s in d:
        convs = s.get("conversations", [])
        if not convs:
            continue
        n_turn = len(convs)
        n_spans = sum(len(t.get("hallucination") or []) for t in convs)
        out[s.get("image_id")] = n_spans / n_turn
    return out


def human_to_record_map(human_path, dyna_json_path):
    H = json.load(open(human_path))
    D = json.load(open(dyna_json_path))

    def fp(s):
        c = s.get("conversations") or [{}]
        return (s.get("image_id"), c[0].get("prompt"))

    d_fp = {fp(s): i for i, s in enumerate(D)}
    out = {}
    for h in H:
        ri = d_fp.get(fp(h))
        if ri is not None:
            out[h.get("image_id")] = ri
    return out


def chair_cov_per_image(grader_dir, model):
    p = find_grader_file(grader_dir, model, "hallucinated_words_{model}.json")
    if not p:
        return {}, {}, {}
    d = json.load(open(p))
    out_chairs, out_chairi, out_cov = {}, {}, {}
    coverage_list = d.get("coverage") or []
    for i, s in enumerate(d.get("sentences", [])):
        img = s.get("image_id")
        m = s.get("metrics", {}) or {}
        if "CHAIRs" in m:
            out_chairs[img] = float(m["CHAIRs"])
        if "CHAIRi" in m:
            out_chairi[img] = float(m["CHAIRi"])
        if i < len(coverage_list):
            out_cov[img] = float(coverage_list[i])
    return out_chairs, out_chairi, out_cov


def mmhal_per_record(examiner, grader_dir, model):
    """Per-record_index hal rate."""
    if (examiner, model) in MMHAL_OVERRIDE:
        sub, fn = MMHAL_OVERRIDE[(examiner, model)]
        p = os.path.join(grader_dir, sub, fn)
        if not os.path.isfile(p):
            p = None
    else:
        p = find_grader_file(grader_dir, model, "mmhal_{model}.json")
    if not p:
        return {}
    d = json.load(open(p))
    bag = defaultdict(list)
    for r in d.get("detailed_results", []):
        bag[r.get("record_index")].append(int(bool(r.get("has_hallucination"))))
    return {ri: sum(v) / len(v) for ri, v in bag.items() if v}


def haelm_per_record(grader_dir, model):
    """HaELM `accurate` is a list of yes/no per atomic claim per turn.
    Per record: average over all turns of (1 - fraction of 'yes')."""
    p = find_grader_file(grader_dir, model, "{model}_haelm.json")
    if not p:
        return {}
    d = json.load(open(p))
    bag = defaultdict(list)
    for ri, sample in enumerate(d):
        for t in sample.get("conversations", []):
            v = t.get("accurate")
            if not v:
                continue
            yes = sum(1 for x in v if str(x).strip().lower().startswith("y"))
            err = 1 - yes / len(v)
            bag[ri].append(err)
    return {ri: sum(v) / len(v) for ri, v in bag.items() if v}


def sg_per_image(grader_dir, model, suffix):
    """sg_ged / sg_delta_con: per-conversation `dist_score`. Aggregate to per
    image_id by mean across that image's conversations."""
    p = find_grader_file(grader_dir, model, "{model}_" + suffix + ".json")
    if not p:
        return {}
    d = json.load(open(p))
    bag = defaultdict(list)
    for s in d:
        sc = s.get("dist_score")
        if sc is None:
            continue
        bag[s.get("image_id")].append(sc)
    return {img: sum(v) / len(v) for img, v in bag.items() if v}


def softspice_per_image(grader_dir, model):
    """softspice: per-sample softspice score from softspice/<model>_<...>.json
    or softspice/spice_<model>.json. Lower = better grounding (higher values
    indicate more spice mismatches)."""
    base = os.path.join(grader_dir, model, "softspice")
    if not os.path.isdir(base):
        base = os.path.join(grader_dir, model, model, "softspice")
    if not os.path.isdir(base):
        return {}
    for fn in os.listdir(base):
        if fn.startswith("spice") and fn.endswith(".json"):
            try:
                d = json.load(open(os.path.join(base, fn)))
            except Exception:
                continue
            entries = d.get("scores") or d.get("results") or d
            if isinstance(entries, dict) and "summary" in entries:
                continue
            if isinstance(entries, list):
                bag = defaultdict(list)
                for s in entries:
                    if not isinstance(s, dict):
                        continue
                    img = s.get("image_id")
                    val = (s.get("softspice") or s.get("score")
                           or s.get("spice") or None)
                    if img is not None and val is not None:
                        bag[img].append(val)
                return {img: sum(v) / len(v) for img, v in bag.items() if v}
    return {}


def faith_per_image(grader_dir, model):
    base = os.path.join(grader_dir, model, "faithscore")
    if not os.path.isdir(base):
        base = os.path.join(grader_dir, model, model, "faithscore")
    if not os.path.isdir(base):
        return {}
    out = {}
    for fn in os.listdir(base):
        if fn.endswith(".json"):
            try:
                d = json.load(open(os.path.join(base, fn)))
            except Exception:
                continue
            if isinstance(d, list):
                for s in d:
                    if not isinstance(s, dict):
                        continue
                    img = s.get("image_id")
                    val = s.get("faithfulness") or s.get("faith") or s.get("score")
                    if img is not None and val is not None:
                        out[img] = val
            elif isinstance(d, dict):
                # try the structure {image_id: score}
                for k, v in d.items():
                    try:
                        out[int(k)] = float(v)
                    except (TypeError, ValueError):
                        pass
    return out


# ---------------------------------------------------------------------------
# Per-examiner pipeline
# ---------------------------------------------------------------------------

GRADER_KEYS = ["chairs", "chairi", "mmhal", "ged", "delcon",
               "softspice", "haelm"]


def collect(human_dir, models, grader_dir, examiner_label):
    rows = []
    cov_by_model = {}
    for human_fn, model in models:
        human_path = os.path.join(human_dir, human_fn)
        dyna_json = find_grader_file(grader_dir, model, "{model}.json")
        if not (os.path.isfile(human_path) and dyna_json):
            print(f"  [{examiner_label}] skip {model}: missing human or dyna")
            continue
        h_per_img = human_per_image(human_path)
        h_to_rec  = human_to_record_map(human_path, dyna_json)
        chairs, chairi, cov = chair_cov_per_image(grader_dir, model)
        mm_rec = mmhal_per_record(examiner_label, grader_dir, model)
        ha_rec = haelm_per_record(grader_dir, model)
        ged   = sg_per_image(grader_dir, model, "sg_ged")
        delc  = sg_per_image(grader_dir, model, "sg_delta_con")
        ssp   = softspice_per_image(grader_dir, model)
        faith = faith_per_image(grader_dir, model)
        cov_by_model[model] = {
            "chairs": len(chairs), "chairi": len(chairi), "cov": len(cov),
            "mmhal": len(mm_rec), "haelm": len(ha_rec),
            "ged": len(ged), "delcon": len(delc), "softspice": len(ssp),
            "faith": len(faith),
        }
        for img, hv in h_per_img.items():
            rec = h_to_rec.get(img)
            rows.append({
                "model": model, "image_id": img, "human": hv,
                "chairs":    chairs.get(img),
                "chairi":    chairi.get(img),
                "1mcov":     (1 - cov[img]) if cov.get(img) is not None else None,
                "mmhal":     mm_rec.get(rec) if rec is not None else None,
                "haelm":     ha_rec.get(rec) if rec is not None else None,
                "ged":       ged.get(img),
                "delcon":    delc.get(img),
                "softspice": ssp.get(img),
                "faith":     faith.get(img),
            })
    print(f"  [{examiner_label}] pooled {len(rows)} per-image rows")
    for m, c in cov_by_model.items():
        print(f"    {m}: {c}")
    return rows


def correlations(rows):
    out = {}
    for k in GRADER_KEYS:
        r, n = pearson([r["human"] for r in rows],
                       [r[k] for r in rows])
        out[k] = (r, n)
    return out


def correlations_per_model(rows):
    """Return {model: {grader: (r, n), ...}, ...}."""
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    out = {}
    for m, sub in by_model.items():
        out[m] = {}
        for k in GRADER_KEYS:
            r, n = pearson([row["human"] for row in sub],
                           [row[k]      for row in sub])
            out[m][k] = (r, n)
    return out


def fmt(r, n):
    if r is None and n >= 3:
        # variance was zero -> Pearson undefined (e.g. CHAIRs saturates to 1)
        return r"sat. \,\scriptsize(%d)" % n
    if r is None:
        return "---"
    return f"${r:+.3f}$ \\,\\scriptsize({n})"


def main():
    print("BASELINE:")
    base_rows = collect(HUMAN_BASELINE, BASELINE_MODELS,
                        BASELINE_GRADER_DIR, "baseline")
    print("CEDI:")
    cedi_rows = collect(HUMAN_CEDI, CEDI_MODELS,
                        CEDI_GRADER_DIR, "CEDI")

    base_corr = correlations(base_rows)
    cedi_corr = correlations(cedi_rows)
    base_pm   = correlations_per_model(base_rows)
    cedi_pm   = correlations_per_model(cedi_rows)

    # CSV
    out_csv = os.path.join(REPO, "plots/human_grader_correlation.csv")
    with open(out_csv, "w") as f:
        f.write("examiner,grader,pearson_r,n\n")
        for ex, corr in [("baseline", base_corr), ("CEDI", cedi_corr)]:
            for k in GRADER_KEYS:
                r, n = corr[k]
                f.write(f"{ex},{k},{'' if r is None else f'{r:.4f}'},{n}\n")
    print(f"[csv] {out_csv}")

    pretty = {
        "chairs": "CHAIRs", "chairi": "CHAIRi",
        "mmhal": "MMHal", "ged": "GED", "delcon": "DELCON",
        "softspice": "SoftSp", "haelm": "HaELM",
    }
    print()
    for ex, corr in [("baseline", base_corr), ("CEDI", cedi_corr)]:
        s = "  ".join(
            f"{pretty[k]} r={r:+.3f} (n={n})"
            if r is not None else f"{pretty[k]}=-- (n={n})"
            for k, (r, n) in corr.items()
        )
        print(f"  {ex:8s} {s}")

    # ---- Per-model dump ----
    out_pm_csv = os.path.join(REPO, "plots/human_grader_correlation_per_model.csv")
    with open(out_pm_csv, "w") as f:
        f.write("examiner,model,grader,pearson_r,n\n")
        for ex, pm in [("baseline", base_pm), ("CEDI", cedi_pm)]:
            for m, by_g in pm.items():
                for k, (r, n) in by_g.items():
                    f.write(f"{ex},{m},{k},"
                            f"{'' if r is None else f'{r:.4f}'},{n}\n")
    print(f"[csv] {out_pm_csv}")
    print()
    for ex, pm in [("baseline", base_pm), ("CEDI", cedi_pm)]:
        print(f"  {ex} (per-model):")
        for m, by_g in pm.items():
            s = "  ".join(
                f"{pretty[k]} r={r:+.3f} (n={n})"
                if r is not None else f"{pretty[k]}=-- (n={n})"
                for k, (r, n) in by_g.items()
            )
            print(f"    {m:28s} {s}")

    # LaTeX
    out_tex = os.path.join(REPO, "plots/human_grader_correlation.tex")
    with open(out_tex, "w") as f:
        header = " & ".join([r"\textbf{" + pretty[k] + "}" for k in GRADER_KEYS])
        f.write(r"""\begin{table*}[h]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Per-image Pearson correlation between human-annotated hallucination
rate and each automatic grader, split by examiner. Sample counts $n$ in
parentheses are images with paired non-missing observations. ``---'' = grader
was not run on the human-annotated models in that examiner setting; ``sat.''
= the grader saturates (zero variance) on this slice and the correlation is
undefined.}
\label{tab:human-grader-corr}
\begin{tabular}{l|""" + "c" * len(GRADER_KEYS) + r"""}
\toprule
\textbf{Examiner} & """ + header + r""" \\
\midrule
""")
        for ex, corr in [("baseline", base_corr), ("CEDI", cedi_corr)]:
            cells = [ex]
            for k in GRADER_KEYS:
                r, n = corr[k]
                cells.append(fmt(r, n))
            f.write(" & ".join(cells) + r" \\" + "\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table*}
""")
    print(f"[tex] {out_tex}")

    # ---- Per-model LaTeX ----
    out_tex_pm = os.path.join(REPO, "plots/human_grader_correlation_per_model.tex")
    pretty_model = {
        "InternVL3-8B-Instruct": "InternVL3-8B",
        "Qwen3-VL-8B-Instruct":  "Qwen3-VL-8B",
        "llava-1.5-7b-hf":        "LLaVA-1.5-7B",
        "opera-llava-1.5":        "Opera-LLaVA-1.5",
    }
    model_order = ["llava-1.5-7b-hf", "InternVL3-8B-Instruct",
                   "Qwen3-VL-8B-Instruct", "opera-llava-1.5"]
    with open(out_tex_pm, "w") as f:
        header = " & ".join([r"\textbf{" + pretty[k] + "}" for k in GRADER_KEYS])
        f.write(r"""\begin{table*}[!htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Per-model Pearson correlation between human-annotated hallucination rate and each automatic grader, broken out by examiner and model. Same conventions as Table~\ref{tab:human-grader-corr}.}
\label{tab:per-model-grader-corr}
\begin{tabular}{ll|""" + "c" * len(GRADER_KEYS) + r"""}
\toprule
\textbf{Examiner} & \textbf{Model} & """ + header + r""" \\
\midrule
""")
        for ex, pm in [("baseline", base_pm), ("CEDI", cedi_pm)]:
            first = True
            for m in model_order:
                if m not in pm:
                    continue
                cells = [ex if first else "", pretty_model.get(m, m)]
                first = False
                for k in GRADER_KEYS:
                    r, n = pm[m][k]
                    cells.append(fmt(r, n))
                f.write(" & ".join(cells) + r" \\" + "\n")
            f.write(r"\midrule" + "\n")
        # remove trailing midrule by replacing with bottomrule
        f.write(r"""\bottomrule
\end{tabular}
\end{table*}
""")
    # actually scrap the trailing extra \midrule\n\bottomrule combo: we wrote
    # two rules. Patch the file in place.
    with open(out_tex_pm) as f:
        body = f.read()
    body = body.replace("\\midrule\n\\bottomrule", "\\bottomrule")
    with open(out_tex_pm, "w") as f:
        f.write(body)
    print(f"[tex] {out_tex_pm}")


if __name__ == "__main__":
    main()
