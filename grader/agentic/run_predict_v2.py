"""Run the v2-agentic annotator over a result file. Same CLI as run_predict.py."""

import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from graders.agentic.annotator_v2 import annotate_round_v2, to_human_format  # noqa: E402


def _iter_rounds(data, limit_images=None):
    for img in data[: limit_images or len(data)]:
        history = []
        for c in img["conversations"]:
            yield img, c, list(history)
            history.append({"prompt": c.get("prompt"), "response": c.get("response"),
                            "q_type": c.get("q_type")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit-images", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-gt", action="store_true")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    tasks = list(_iter_rounds(data, args.limit_images))
    print(f"[v2] Annotating {len(tasks)} rounds, workers={args.workers}")

    def work(item):
        img, c, history = item
        gt = None if args.no_gt else c.get("gt")
        spans = annotate_round_v2(
            image=img, prompt=c.get("prompt", ""), response=c.get("response", ""),
            q_type=c.get("q_type", ""), history=history, gt=gt,
            round_id=c.get("round_id"), use_cache=not args.no_cache,
        )
        return c, spans

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for fut in as_completed(futs):
            c, spans = fut.result()
            c["hallucination"] = to_human_format(spans)
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data[: args.limit_images or len(data)], f, indent=2)
    print(f"Wrote v2 predictions -> {args.output}")


if __name__ == "__main__":
    main()
