"""Dev harness: run the annotator on a small balanced subset of the two
human-annotated files and report agreement, so the prompt can be iterated
cheaply (results are cached on disk, keyed by PROMPT_VERSION).

Usage:
  python -m graders.agentic.dev_tune [--limit-images 8] [--workers 8]
"""

import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from graders.agentic.annotator import annotate_round, to_human_format, PROMPT_VERSION  # noqa: E402
from graders.agentic.evaluate import evaluate  # noqa: E402

GOLD_DIR = "work_dirs/human/result/dyna-v18"
DEV_FILES = [
    "opera-llava-1.5_cache_first50.json",
    "llava-1.5-7b-hf_single_first50.json",
]


def predict_inplace(data, limit_images, workers):
    tasks = []
    for img in data[:limit_images]:
        history = []
        for c in img["conversations"]:
            tasks.append((img, c, list(history)))
            history.append({"prompt": c.get("prompt"), "response": c.get("response"),
                            "q_type": c.get("q_type")})

    def work(item):
        img, c, history = item
        spans = annotate_round(
            image_id=img["image_id"], url=img["url"],
            prompt=c.get("prompt", ""), response=c.get("response", ""),
            q_type=c.get("q_type", ""), history=history, gt=c.get("gt"),
            round_id=c.get("round_id"),
        )
        return c, spans

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for fut in as_completed(futs):
            c, spans = fut.result()
            c["hallucination_pred"] = to_human_format(spans)
    # move pred into hallucination slot of a copy for evaluate()
    return len(tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-images", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    print(f"PROMPT_VERSION={PROMPT_VERSION}  limit_images={args.limit_images}")
    for fn in DEV_FILES:
        path = os.path.join(GOLD_DIR, fn)
        with open(path) as f:
            gold = json.load(f)
        n = predict_inplace(gold, args.limit_images, args.workers)
        # build a pred structure where hallucination = predicted
        import copy
        pred = copy.deepcopy(gold)
        for img in pred[: args.limit_images]:
            for c in img["conversations"]:
                c["hallucination"] = c.get("hallucination_pred", [])
        report = evaluate(gold, pred, args.limit_images)
        print(f"\n===== {fn}  ({n} rounds) =====")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
