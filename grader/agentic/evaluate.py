"""Evaluate agentic predictions against human gold annotations.

Reports two levels of agreement:

  * Round-level DETECTION: did we flag the round as hallucinated (>=1 span) vs the
    human? -> precision / recall / F1 / accuracy over rounds (and per q_type).
  * SPAN-level overlap: greedy char-level matching between predicted and gold spans
    -> mean best-IoU per gold span, and match rate at IoU>=0.5; plus token-F1.

Gold spans with empty / whitespace text are treated as "no hallucination".
Non-substring gold spans (rare human typos) still count via char overlap.

Usage:
  python -m graders.agentic.evaluate --gold GOLD.json --pred PRED.json [--limit-images N]
"""

import os
import sys
import json
import argparse
from collections import defaultdict


def _gold_spans(conv):
    return [h["hallucination"] for h in (conv.get("hallucination") or [])
            if isinstance(h, dict) and h.get("hallucination", "").strip()]


def _pred_spans(conv):
    out = []
    for h in (conv.get("hallucination") or []):
        if isinstance(h, dict):
            s = h.get("hallucination", "")
        else:
            s = h
        if isinstance(s, str) and s.strip():
            out.append(s)
    return out


def _char_set(response, span):
    """Set of char indices in response covered by the first occurrence of span."""
    low_r, low_s = response.lower(), span.lower()
    i = low_r.find(low_s)
    if i < 0:
        # fall back to longest matching block
        import difflib
        m = difflib.SequenceMatcher(None, low_s, low_r, autojunk=False).find_longest_match(
            0, len(low_s), 0, len(low_r))
        if m.size == 0:
            return set()
        return set(range(m.b, m.b + m.size))
    return set(range(i, i + len(span)))


def _iou(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def evaluate(gold_data, pred_data, limit_images=None):
    # index pred by (image_id, round_id)
    pred_idx = {}
    for img in pred_data:
        for c in img["conversations"]:
            pred_idx[(img["image_id"], c["round_id"])] = c

    # detection counters
    tp = fp = fn = tn = 0
    by_qt = defaultdict(lambda: dict(tp=0, fp=0, fn=0, tn=0))
    # span counters
    iou_sum = 0.0
    iou_n = 0
    matched_05 = 0
    tok_tp = tok_fp = tok_fn = 0
    missing = 0

    for img in gold_data[: limit_images or len(gold_data)]:
        resp_by_round = {c["round_id"]: c.get("response", "") for c in img["conversations"]}
        for c in img["conversations"]:
            key = (img["image_id"], c["round_id"])
            pc = pred_idx.get(key)
            if pc is None:
                missing += 1
                continue
            response = resp_by_round[c["round_id"]]
            g = _gold_spans(c)
            p = _pred_spans(pc)
            qt = c.get("q_type", "?")

            g_has, p_has = bool(g), bool(p)
            if g_has and p_has:
                tp += 1; by_qt[qt]["tp"] += 1
            elif not g_has and p_has:
                fp += 1; by_qt[qt]["fp"] += 1
            elif g_has and not p_has:
                fn += 1; by_qt[qt]["fn"] += 1
            else:
                tn += 1; by_qt[qt]["tn"] += 1

            # span overlap (only meaningful when gold has spans)
            g_sets = [_char_set(response, s) for s in g]
            p_sets = [_char_set(response, s) for s in p]
            for gs in g_sets:
                best = max((_iou(gs, ps) for ps in p_sets), default=0.0)
                iou_sum += best
                iou_n += 1
                if best >= 0.5:
                    matched_05 += 1
            # token-level (char) F1 over the union of spans
            g_union = set().union(*g_sets) if g_sets else set()
            p_union = set().union(*p_sets) if p_sets else set()
            tok_tp += len(g_union & p_union)
            tok_fp += len(p_union - g_union)
            tok_fn += len(g_union - p_union)

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    det_p, det_r, det_f = prf(tp, fp, fn)
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    tok_p, tok_r, tok_f = prf(tok_tp, tok_fp, tok_fn)

    report = {
        "rounds_scored": n,
        "missing_pred_rounds": missing,
        "detection": {
            "precision": round(det_p, 3), "recall": round(det_r, 3),
            "f1": round(det_f, 3), "accuracy": round(acc, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "detection_by_qtype": {},
        "span": {
            "mean_best_iou_per_gold_span": round(iou_sum / iou_n, 3) if iou_n else None,
            "gold_spans": iou_n,
            "match_rate_iou>=0.5": round(matched_05 / iou_n, 3) if iou_n else None,
            "char_token_f1": round(tok_f, 3),
            "char_token_precision": round(tok_p, 3),
            "char_token_recall": round(tok_r, 3),
        },
    }
    for qt, d in by_qt.items():
        p, r, f = prf(d["tp"], d["fp"], d["fn"])
        nn = sum(d.values())
        report["detection_by_qtype"][qt] = {
            "n": nn, "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(f, 3),
            "accuracy": round((d["tp"] + d["tn"]) / nn, 3) if nn else 0.0,
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--limit-images", type=int, default=None)
    args = ap.parse_args()
    with open(args.gold) as f:
        gold = json.load(f)
    with open(args.pred) as f:
        pred = json.load(f)
    report = evaluate(gold, pred, args.limit_images)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
