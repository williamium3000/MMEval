"""Per-turn VALOR object-existence grader.

Reads a per-turn CHAIR output (`sentences` list with `round_id`, `q_type`,
`vg_gt_words`, `vg_generated_words`) and, for each turn, calls the judge
LLM to match generated objects against the ground-truth object set with
broader-concept tolerance. Then computes per-turn faith_i and cov_i and
aggregates by q_type + overall.

Runs LLM calls concurrently via ThreadPoolExecutor; the underlying HTTP
client (from `grader.valor.gpt_model`) is thread-safe.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from grader.valor.gpt_model import llm  # noqa: E402


PROMPT_MATCH = """
You are given a list of ground-truth objects for an image and a list of
objects the model actually mentioned in one turn of dialogue. Determine
which mentioned objects can be matched to a ground-truth object either
exactly, via a synonym/hyponym, or because the GT object is a broader
concept that includes the mentioned one.

Reply ONLY with a JSON object of the form:
```json
{{
    "matched_objects": {{"<generated_object>": "<gt_object>", ...}},
    "broader_concept":  {{"<generated_object>": "<gt_object>", ...}}
}}
```

- `matched_objects`: generated objects that map to a specific GT object.
- `broader_concept`: generated objects that fall under a broader GT
  concept (e.g. GT "furniture" covers generated "chair").
- Do NOT hallucinate GT objects that are not in the input list.
- Do NOT include a generated object in both dicts.
- If nothing matches, both dicts should be empty.

Ground-truth objects: [GT_OBJECTS]
Generated (this turn): [GENERATED_OBJECTS]
"""


def _match_one(sent, model):
    gt = sorted({str(w) for w in (sent.get("vg_gt_words") or []) if w})
    gen = sorted({str(w) for w in (sent.get("vg_generated_words") or []) if w})
    if not gen:
        return {
            "matched_objects": {},
            "broader_concept":  {},
        }
    prompt = PROMPT_MATCH.replace("[GT_OBJECTS]", str(gt)).replace(
        "[GENERATED_OBJECTS]", str(gen)
    )
    out = llm([{"role": "user", "content": prompt}], model=model)
    matched = out.get("matched_objects", {}) or {}
    broader = out.get("broader_concept",  {}) or {}
    # Guard against non-dict return shapes
    if not isinstance(matched, dict): matched = {}
    if not isinstance(broader, dict): broader = {}
    return {"matched_objects": matched, "broader_concept": broader}


def _per_turn_metrics(gt, gen, matched, broader):
    gen_set = set(gen)
    if not gen_set:
        faith_i = 0.0
    else:
        matched_count = len(set(matched.keys()) & gen_set) + len(set(broader.keys()) & gen_set)
        faith_i = matched_count / len(gen_set)
    gt_set = set(gt)
    if not gt_set:
        cov_i = 0.0
    else:
        matched_gt = set(matched.values())
        cov_i = len(matched_gt & gt_set) / len(gt_set)
    return faith_i, cov_i


def _flush(out_sents, output_path):
    """Write partial results so a killed run can resume from this state."""
    keep = [x for x in out_sents if x is not None]
    if not keep:
        return
    with open(output_path, "w") as fh:
        json.dump({"sentences": keep, "partial": True}, fh)


def run(input_path, output_path, model, max_workers, sample_num, flush_interval=500):
    d = json.load(open(input_path))
    sents = d.get("sentences", d) if isinstance(d, dict) else d
    if not sents:
        raise ValueError(f"empty input: {input_path}")

    # Require per-turn shape
    if "round_id" not in sents[0] or "q_type" not in sents[0]:
        raise ValueError(
            f"{input_path}: sentences missing round_id/q_type — "
            f"input must be a --by_qtype CHAIR output"
        )

    sents = sents[:sample_num] if sample_num and sample_num > 0 else sents

    # Resume: if a partial output exists, merge already-computed matches.
    already = {}
    if os.path.isfile(output_path):
        try:
            prior = json.load(open(output_path))
            for s in prior.get("sentences", []):
                key = (s.get("image_id"), s.get("round_id"))
                if s.get("matched_objects") is not None:
                    already[key] = (s.get("matched_objects"), s.get("broader_concept", {}))
        except Exception:
            pass

    out_sents = [None] * len(sents)

    def _task(i, s):
        key = (s.get("image_id"), int(s.get("round_id") or 0))
        if key in already:
            m, b = already[key]
        else:
            r = _match_one(s, model)
            m, b = r["matched_objects"], r["broader_concept"]
        gt = sorted({str(w) for w in (s.get("vg_gt_words") or []) if w})
        gen = sorted({str(w) for w in (s.get("vg_generated_words") or []) if w})
        faith, cov = _per_turn_metrics(gt, gen, m, b)
        return i, {
            "image_id":         s.get("image_id"),
            "round_id":         int(s.get("round_id") or 0),
            "q_type":           s.get("q_type"),
            "gt_words":         gt,
            "generated_words":  gen,
            "matched_objects":  m,
            "broader_concept":  b,
            "metrics": {
                "faithfulness_score_i": faith,
                "coverage_score_i":     cov,
            },
        }

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_task, i, s) for i, s in enumerate(sents)]
        for fut in tqdm.tqdm(as_completed(futs), total=len(futs), desc=os.path.basename(input_path)):
            i, item = fut.result()
            out_sents[i] = item
            completed += 1
            if flush_interval and completed % flush_interval == 0:
                _flush(out_sents, output_path)

    # aggregate by q_type + overall
    per_qt = defaultdict(list)  # qt -> [(faith, cov, n_gen, n_gt)]
    for item in out_sents:
        faith = item["metrics"]["faithfulness_score_i"]
        cov   = item["metrics"]["coverage_score_i"]
        per_qt[item["q_type"]].append((faith, cov, len(item["generated_words"]),
                                       len(item["gt_words"])))

    def _agg(rows):
        if not rows:
            return 0., 0., 0
        faith = sum(r[0] for r in rows) / len(rows)
        cov   = sum(r[1] for r in rows) / len(rows)
        return faith, cov, len(rows)

    overall_rows = [r for rows in per_qt.values() for r in rows]
    f, c, n = _agg(overall_rows)
    output = {
        "sentences": out_sents,
        "overall_metrics": {
            "faithfulness_score_i": f,
            "coverage_score_i":     c,
            "n_turns":              n,
        },
        "by_qtype": {qt: dict(zip(["faithfulness_score_i","coverage_score_i","n_turns"], _agg(rows)))
                       for qt, rows in per_qt.items()},
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"[{input_path}] overall faith_i={f*100:.2f}%, cov_i={c*100:.2f}%, n={n}")
    for qt, m in output["by_qtype"].items():
        print(f"  {qt:14s}  faith_i={m['faithfulness_score_i']*100:5.2f}%  "
              f"cov_i={m['coverage_score_i']*100:5.2f}%  n={m['n_turns']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-ip", "--input_path",  required=True,
                   help="path to a --by_qtype CHAIR output "
                        "(hallucinated_words_*.json with round_id/q_type)")
    p.add_argument("-op", "--output_path", required=True)
    p.add_argument("--model", default=None,
                   help="LLM name (defaults to VALOR_MODEL env, else Qwen3-30B)")
    p.add_argument("--max_workers", type=int, default=32)
    p.add_argument("--sample_num",  type=int, default=0,
                   help="0 = use all sentences")
    p.add_argument("--flush_interval", type=int, default=500,
                   help="write partial output after every N completions "
                        "(so a killed run can resume). 0 = only at end.")
    args = p.parse_args()
    run(args.input_path, args.output_path, args.model,
        args.max_workers, args.sample_num, args.flush_interval)
