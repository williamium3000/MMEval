"""CHAIR-i v2 (object-scoped, per-turn) variant of Fig 11.

Signal per turn = fraction of the model's own object mentions that are
hallucinated, deduplicated within the turn:

    CHAIRi_v2(turn) = |unique hallucinated vg objects in turn|
                   / |unique vg objects mentioned in turn|

Denominator is scoped to VG-matched objects only (not raw tokens),
so this is directly comparable to the aggregate CHAIRi_v2 reported
in the CHAIR summary CSVs.

Reuses the 3-panel layout from error_analysis_pope_human. Panel (a)
still counts turns with any hallucinated object; panels (b/c) show
mean per-turn CHAIRi_v2 rather than a binary rate.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from error_analysis import Q_TYPES, REPO, rate_by_qtype
from error_analysis_pope_human import _plot_3panel


V18_DIR      = os.path.join(REPO, "work_dirs/vg/final_run_v18_gpt4o_completed")
V18CONV_DIR  = os.path.join(REPO, "work_dirs/vg/final_run_v18_gpt4o_conv_completed")
OUT_PDF      = os.path.join(REPO, "nips_paper/fig/error_analysis_chair_v2.pdf")

EXCLUDE_PREFIX = ("InternVL", "internvl")


def _candidate_paths(base_dir, model):
    return [
        os.path.join(base_dir, model, f"{model}_chair_results.json"),
        os.path.join(base_dir, f"hallucinated_words_{model}.json"),
        os.path.join(base_dir, model, f"hallucinated_words_{model}.json"),
    ]


def _load_chair(base_dir, model):
    for p in _candidate_paths(base_dir, model):
        if not os.path.isfile(p):
            continue
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  skip {p}: {e}")
            continue
        sents = d.get("sentences", [])
        if not sents:
            continue
        if "round_id" not in sents[0] or "q_type" not in sents[0]:
            continue
        return p, sents
    return None, None


def _dedup_key(item):
    if isinstance(item, dict):
        return item.get("synset") or item.get("word")
    if isinstance(item, (list, tuple)) and item:
        return item[1] if len(item) > 1 else item[0]
    return item


def _per_sentence_ratio(sent):
    """Return (n_generated, n_hallucinated) for one sentence, deduped."""
    # Format 1: positive_claims / false_positives (list of {word, synset})
    # Format 2: vg_generated_words (list[str]) / vg_hallucinated_words (list[tuple])
    gens = sent.get("positive_claims") or sent.get("vg_generated_words") or []
    halls = sent.get("false_positives") or sent.get("vg_hallucinated_words") or []
    gen_keys  = {_dedup_key(g) for g in gens  if _dedup_key(g) is not None}
    hall_keys = {_dedup_key(h) for h in halls if _dedup_key(h) is not None}
    return len(gen_keys), len(hall_keys)


def _collect(base_dir, label):
    rows, used = [], []
    if not os.path.isdir(base_dir):
        return rows, used
    seen_models = set()
    for entry in sorted(os.listdir(base_dir)):
        if any(s in entry for s in ("_extracted", "_pope", "_summary",
                                     "_with_both", "_sg", ".log",
                                     "results_", "reference")):
            continue
        m = (entry.replace("hallucinated_words_", "").replace(".json", "")
             if entry.startswith("hallucinated_words_") else entry)
        if m in seen_models:
            continue
        if m.startswith(EXCLUDE_PREFIX):
            continue
        p, sents = _load_chair(base_dir, m)
        if not sents:
            continue
        seen_models.add(m)
        used.append(m)
        max_round_per_image = {}
        for s in sents:
            imid = s.get("image_id")
            r = int(s.get("round_id") or 0)
            if r > max_round_per_image.get(imid, 0):
                max_round_per_image[imid] = r
        for s in sents:
            qt = s.get("q_type")
            if qt not in Q_TYPES:
                continue
            n_gen, n_hall = _per_sentence_ratio(s)
            if n_gen == 0:
                continue
            r = int(s.get("round_id") or 0)
            mx = max_round_per_image.get(s.get("image_id"), 0) or 1
            rows.append((m, qt, r / mx, n_hall / n_gen))
    print(f"[{label}] {len(used)} models, {len(rows)} per-turn records")
    return rows, sorted(used)


def main():
    v18_rows,  v18_models  = _collect(V18_DIR,     "CHAIRi_v2 v18")
    conv_rows, conv_models = _collect(V18CONV_DIR, "CHAIRi_v2 v18conv")
    keep = sorted(set(v18_models) & set(conv_models))
    if not keep:
        print("No overlapping evaluatees with per-turn CHAIR data.")
        print(f"  v18:     {v18_models}")
        print(f"  v18conv: {conv_models}")
        return
    v18_rows  = [r for r in v18_rows  if r[0] in keep]
    conv_rows = [r for r in conv_rows if r[0] in keep]
    print(f"Paired evaluatees ({len(keep)}): {keep}")

    series = {
        "CHAIR$_i$ v2 -- no history":   v18_rows,
        "CHAIR$_i$ v2 -- with history": conv_rows,
    }

    _plot_3panel(
        series,
        out_pdf=OUT_PDF,
        source_label="CHAIR$_i$ v2",
        pie_series_key="CHAIR$_i$ v2 -- no history",
        palette={
            "CHAIR$_i$ v2 -- no history":   "#D5E4F4",
            "CHAIR$_i$ v2 -- with history": "#E4F0E0",
        },
        edge={
            "CHAIR$_i$ v2 -- no history":   "#2F6FCC",
            "CHAIR$_i$ v2 -- with history": "#496F2C",
        },
        markers={
            "CHAIR$_i$ v2 -- no history":   "o",
            "CHAIR$_i$ v2 -- with history": "D",
        },
    )
    for k in series:
        agg = rate_by_qtype(series[k])
        s = " | ".join(f"{qt}={agg[qt][0]*100:5.1f}% (n={agg[qt][1]})"
                        for qt in Q_TYPES)
        print(f"  {k:32s}  {s}")


if __name__ == "__main__":
    main()
