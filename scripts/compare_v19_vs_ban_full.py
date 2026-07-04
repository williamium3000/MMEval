"""Aggregate every grader output we have for the 3 VG models and compare
v19 (the non-ban2type baseline) vs v19ban2type. Produces a flat comparison
table on stdout + a JSON blob at work_dirs/vg/_full_comparison.json.

Graders covered:
  - agentic   (grader/agentic/predictions/vg_{side}/{model}.pred.json)
  - mmhal     (work_dirs/vg/{side}/{model}/mmhal/mmhal_{model}.json)
  - valor obj_exist  (..valor/{model}_obj_exist.json)
  - valor rel_pos    (..valor/{model}_rel_pos.json)
  - chair     (..chair/{model}_chair_metrics.txt) — parsed from metric printouts
  - sg/delta_con (..sg/{model}_sg_delta_con.json)
"""
import glob
import json
import os
import re
from collections import defaultdict


MODELS = ["gemma-3-12b-it", "llava-1.5-7b-hf", "Qwen2.5-VL-7B-Instruct"]
SIDES = ["v19", "v19ban2type"]


def agentic_rate(side, model):
    p = f"grader/agentic/predictions/vg_{side}/{model}.pred.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    rounds = halluc = 0
    for img in d:
        for c in img.get("conversations", []):
            rounds += 1
            if c.get("hallucination"):
                halluc += 1
    return (halluc / rounds * 100) if rounds else None


def mmhal_metric(side, model):
    p = f"work_dirs/vg/{side}/{model}/mmhal/mmhal_{model}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    o = d.get("overall_metrics", {})
    return {
        "avg_score": o.get("avg_score"),
        "hall_rate": o.get("hallucination_rate"),
        "n": o.get("total_evaluations"),
    }


def valor_metric(side, model, sub):
    p = f"work_dirs/vg/{side}/{model}/valor/{model}_{sub}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d.get("overall_metrics", {})


def chair_metric(side, model):
    """Parse the per-q_type CHAIR table and compute sentence-weighted averages.

    The script's printed "overall" row is a SUM (not a mean) so we ignore it
    and weight each q_type row by its Sentences column. Also returns the raw
    Hallucinated/Sentences totals → a sentence-level hallucination rate.
    """
    p = f"work_dirs/vg/{side}/{model}/chair/{model}_chair_metrics.txt"
    if not os.path.exists(p):
        return None
    rows = []
    for line in open(p):
        if not line[:1].isalpha() or line.startswith(("Question-Type", "overall", "CHAIRs")):
            continue
        parts = line.split()
        # adversarial 35.5 6.8 18.5 4.8 22.3 386 137
        if len(parts) < 8:
            continue
        try:
            rows.append({
                "q": parts[0],
                "CHAIRs": float(parts[1]),
                "CHAIRi": float(parts[2]),
                "CHAIRi_v2": float(parts[3]),
                "Cov_avg": float(parts[4]),
                "Cov_all": float(parts[5]),
                "Sent": int(parts[6]),
                "Hall": int(parts[7]),
            })
        except ValueError:
            continue
    if not rows:
        return None
    tot_sent = sum(r["Sent"] for r in rows)
    tot_hall = sum(r["Hall"] for r in rows)
    out = {
        "n_sentences": tot_sent,
        "n_hallucinated": tot_hall,
        "sentence_hall_rate": tot_hall / tot_sent * 100 if tot_sent else None,
    }
    for k in ("CHAIRs", "CHAIRi", "CHAIRi_v2", "Cov_avg", "Cov_all"):
        out[k] = sum(r[k] * r["Sent"] for r in rows) / tot_sent if tot_sent else None
    return out


def sg_metric(side, model):
    p = f"work_dirs/vg/{side}/{model}/sg/{model}_sg_delta_con.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    # graph_distance output is a list of per-sample dicts with a `dist_score` field.
    distances = []
    if isinstance(d, list):
        for x in d:
            if not isinstance(x, dict):
                continue
            v = x.get("dist_score")
            if isinstance(v, (int, float)):
                distances.append(v)
    if not distances:
        return None
    return {"avg": sum(distances) / len(distances), "n": len(distances)}


def fmt(v, kind="num"):
    if v is None:
        return "  n/a "
    if kind == "pct":
        return f"{v:5.1f}%"
    if kind == "num4":
        return f"{v:6.3f}"
    return f"{v:6.2f}"


def main():
    out = {}
    for m in MODELS:
        out[m] = {}
        for s in SIDES:
            out[m][s] = {
                "agentic_rate":   agentic_rate(s, m),
                "mmhal":          mmhal_metric(s, m),
                "valor_obj_exist": valor_metric(s, m, "obj_exist"),
                "valor_rel_pos":  valor_metric(s, m, "rel_pos"),
                "chair":          chair_metric(s, m),
                "sg_delta_con":   sg_metric(s, m),
            }

    # Pretty print: one section per metric, table with model rows and 2 side cols + delta
    sections = [
        ("Agentic hallucination rate (%, lower better)",
         lambda r: r["agentic_rate"], "pct"),
        ("MMHal avg score (1-6, higher better)",
         lambda r: r["mmhal"]["avg_score"] if r["mmhal"] else None, "num"),
        ("MMHal hallucination rate (%, lower better)",
         lambda r: r["mmhal"]["hall_rate"] * 100 if r["mmhal"] else None, "pct"),
        ("VALOR object-existence faithfulness_i (higher better)",
         lambda r: r["valor_obj_exist"]["faithfulness_score_i"] if r["valor_obj_exist"] else None, "num4"),
        ("VALOR object-existence coverage_i (higher better)",
         lambda r: r["valor_obj_exist"]["coverage_score_i"] if r["valor_obj_exist"] else None, "num4"),
        ("VALOR relation-positional faithfulness_i (higher better)",
         lambda r: r["valor_rel_pos"]["faithfulness_score_i"] if r["valor_rel_pos"] else None, "num4"),
        ("CHAIR sentence-level hall rate (%, lower better)",
         lambda r: r["chair"]["sentence_hall_rate"] if r["chair"] else None, "pct"),
        ("CHAIR CHAIRs (sentence-weighted, lower better)",
         lambda r: r["chair"]["CHAIRs"] if r["chair"] else None, "num"),
        ("CHAIR CHAIRi (sentence-weighted, lower better)",
         lambda r: r["chair"]["CHAIRi"] if r["chair"] else None, "num"),
        ("CHAIR Cov_all (sentence-weighted, higher better)",
         lambda r: r["chair"]["Cov_all"] if r["chair"] else None, "num"),
        ("SG DELCON distance (lower = closer to GT scene graph)",
         lambda r: r["sg_delta_con"]["avg"] if r["sg_delta_con"] else None, "num4"),
    ]

    print(f"\n{'='*86}")
    print("v19 vs v19ban2type — all completed graders, VG, 25 samples × 3 models")
    print(f"{'='*86}\n")
    for title, extract, kind in sections:
        print(f"--- {title} ---")
        print(f"{'Model':<26} {'v19':>10} {'v19ban':>10} {'delta(ban-v19)':>16}")
        for m in MODELS:
            v19v   = extract(out[m]["v19"])
            v19bv  = extract(out[m]["v19ban2type"])
            if isinstance(v19v, (int, float)) and isinstance(v19bv, (int, float)):
                delta = v19bv - v19v
                delta_str = (f"{delta:+6.3f}" if kind == "num4"
                             else f"{delta:+6.2f}" if kind == "num"
                             else f"{delta:+5.1f}pp")
            else:
                delta_str = "    n/a"
            print(f"{m:<26} {fmt(v19v, kind):>10} {fmt(v19bv, kind):>10} {delta_str:>16}")
        print()

    os.makedirs("work_dirs/vg", exist_ok=True)
    json.dump(out, open("work_dirs/vg/_full_comparison.json", "w"), indent=2, default=str)
    print(f"\nwrote work_dirs/vg/_full_comparison.json")


if __name__ == "__main__":
    main()
