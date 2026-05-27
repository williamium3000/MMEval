"""Run the agentic annotator over a dyna-v18 result file and write predictions.

Each round's predicted hallucination spans are written into a ``hallucination``
field (human-file schema), preserving all other fields. Calls run concurrently.

Usage:
  python -m graders.agentic.run_predict \
      --input  work_dirs/vg/final_run_v18_gpt4o_completed/InternVL3-8B-Instruct/InternVL3-8B-Instruct.json \
      --output graders/agentic/predictions/InternVL3-8B-Instruct.pred.json \
      [--limit-images N] [--workers 8]
"""

import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from graders.agentic.annotator import annotate_round, to_human_format  # noqa: E402


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
    ap.add_argument("--no-gt", action="store_true",
                    help="Ignore the per-round gt field even if present.")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    tasks = list(_iter_rounds(data, args.limit_images))
    print(f"Annotating {len(tasks)} rounds from {args.input} "
          f"({min(len(data), args.limit_images or len(data))} images), workers={args.workers}")

    def work(item):
        img, c, history = item
        gt = None if args.no_gt else c.get("gt")
        spans = annotate_round(
            image_id=img["image_id"], url=img["url"],
            prompt=c.get("prompt", ""), response=c.get("response", ""),
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
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)} rounds done")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data[: args.limit_images or len(data)], f, indent=2)
    print(f"Wrote predictions -> {args.output}")


if __name__ == "__main__":
    main()
