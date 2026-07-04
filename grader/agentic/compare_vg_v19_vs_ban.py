"""Compare agentic hallucination rates for v19 vs v19ban2type on VG for the
3 shared models (gemma-3-12b-it, llava-1.5-7b-hf, Qwen2.5-VL-7B-Instruct).

For each (model, side):
  - flat hallucination rate = #(rounds with >=1 high-conf span) / #rounds
  - by q_type breakdown (regular, follow-up, adversarial, unanswerable)

Only counts rounds for image_ids present in BOTH sides per model (apples-to-apples).
"""
import json
import os
from collections import defaultdict

MODELS = ["gemma-3-12b-it", "llava-1.5-7b-hf", "Qwen2.5-VL-7B-Instruct"]
SIDES = ["v19", "v19ban2type"]

QTYPES = ["regular", "follow-up", "adversarial", "unanswerable"]


def load(side, model):
    p = f"grader/agentic/predictions/vg_{side}/{model}.pred.json"
    return json.load(open(p))


def common_image_ids(model):
    sets = []
    for s in SIDES:
        d = load(s, model)
        sets.append({x["image_id"] for x in d})
    return sets[0] & sets[1]


def stats(side, model, allowed_ids):
    d = load(side, model)
    n_rounds = 0
    n_hall = 0
    by_q = defaultdict(lambda: [0, 0])    # [hall, total]
    for img in d:
        if img["image_id"] not in allowed_ids:
            continue
        for c in img.get("conversations", []):
            n_rounds += 1
            q = c.get("q_type", "?")
            hallucinated = bool(c.get("hallucination") or [])
            by_q[q][1] += 1
            if hallucinated:
                n_hall += 1
                by_q[q][0] += 1
    rate = (n_hall / n_rounds * 100) if n_rounds else 0.0
    by_q_rate = {q: (h/t*100 if t else 0.0, t) for q, (h, t) in by_q.items()}
    return {"n_rounds": n_rounds, "n_hall": n_hall, "rate": rate, "by_q": by_q_rate}


def main():
    print(f"\n{'='*78}")
    print(f"Agentic hallucination rate (threshold=high) - v19 vs v19ban2type on VG")
    print(f"{'='*78}\n")
    print(f"{'Model':<26} {'side':<12} {'n_img':>6} {'rounds':>7} {'#hall':>6} {'rate':>7}")
    print(f"{'-'*78}")
    all_rows = {}
    for m in MODELS:
        common = common_image_ids(m)
        row = {}
        for s in SIDES:
            st = stats(s, m, common)
            row[s] = st
            print(f"{m:<26} {s:<12} {len(common):>6} {st['n_rounds']:>7} {st['n_hall']:>6} {st['rate']:>6.1f}%")
        delta = row["v19ban2type"]["rate"] - row["v19"]["rate"]
        print(f"{'  delta (ban - v19)':<26} {'':<12} {'':>6} {'':>7} {'':>6} {delta:>+6.1f}pp")
        all_rows[m] = row
        print()

    print(f"\n{'='*78}")
    print("By q_type breakdown (hall rate %, n_rounds)")
    print(f"{'='*78}\n")
    for m in MODELS:
        print(f"--- {m} ---")
        print(f"{'q_type':<14}", *[f"{s:>20}" for s in SIDES])
        for q in QTYPES:
            line = [f"{q:<14}"]
            for s in SIDES:
                if q in all_rows[m][s]["by_q"]:
                    r, t = all_rows[m][s]["by_q"][q]
                    line.append(f"{r:>5.1f}% ({t:>4})    ")
                else:
                    line.append(f"{'n/a':>20}")
            print(*line)
        print()

    # also dump JSON for the record
    out = "grader/agentic/predictions/vg_comparison.json"
    json.dump({m: {s: {k: v for k, v in d.items() if k != 'by_q'} | {
        'by_q': {q: {'rate': r[0], 'n': r[1]} for q, r in d['by_q'].items()}
    } for s, d in row.items()} for m, row in all_rows.items()}, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
