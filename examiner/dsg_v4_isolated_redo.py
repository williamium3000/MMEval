"""Re-run DSG_v4 single-VLM step with ONE call per POPE question (no batching).

Adds new fields next to existing ones, never overwrites:
  qa["single_response_isolated"]      -> "Yes"/"No"/"Unknown"
  qa["single_response_isolated_raw"]  -> full gpt-4o reply for that single question
  qa["single_response_isolated_correct"] (when ref available)

Designed to be a side-by-side cross-check against the per-turn-batched
single_response, NOT a replacement.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

# Ensure project root on path
_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from examiner.DSG_v4 import call_vlm_api, parse_yesno_list


# get_image_path is defined inside main() in DSG_v4; replicate here.
def get_image_path(sample):
    return sample.get("image") or sample.get("url") or sample.get("image_path")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--num_samples", type=int, default=10)
    ap.add_argument("--vlm_api_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.uniapi.io/v1"))
    ap.add_argument("--vlm_api_key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--vlm_api_model", default="gpt-4o")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    if not args.vlm_api_key:
        sys.exit("Need OPENAI_API_KEY (or --vlm_api_key)")

    with open(args.input_file, "r") as f:
        data = json.load(f)

    # Build a flat list of (sample_idx, turn_idx, qa_idx, question, image_path) for first N samples
    work = []
    for si, sample in enumerate(data[: args.num_samples]):
        image_path = get_image_path(sample)
        if not image_path:
            continue
        for ti, turn in enumerate(sample.get("conversations", [])):
            for qi, qa in enumerate(turn.get("dsg_qa", [])):
                q = qa.get("question", "")
                if not q:
                    continue
                work.append((si, ti, qi, q, image_path))
    print(f"Total isolated VLM calls to make: {len(work)}")

    def ask_one(item):
        si, ti, qi, q, image_path = item
        prompt = f"Look at the image and answer the following question with ONLY Yes or No. Write nothing else.\n\nQuestion: {q}\n\nAnswer:"
        try:
            raw = call_vlm_api(args.vlm_api_url, image_path, prompt, args.vlm_api_model, api_key=args.vlm_api_key)
        except Exception as e:
            raw = f"__ERROR__ {type(e).__name__}: {e}"
        return (si, ti, qi, raw)

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(ask_one, w) for w in work]
        done = 0
        for f in as_completed(futures):
            si, ti, qi, raw = f.result()
            results[(si, ti, qi)] = raw
            done += 1
            if done % 50 == 0 or done == len(work):
                print(f"  {done}/{len(work)} done")

    # Patch results back in
    for (si, ti, qi), raw in results.items():
        qa = data[si]["conversations"][ti]["dsg_qa"][qi]
        # Single-question parse: parse_yesno_list with n_expected=1
        parsed = parse_yesno_list(raw, 1)
        qa["single_response_isolated"] = parsed[0] if parsed else "Unknown"
        qa["single_response_isolated_raw"] = raw

    with open(args.outfile, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote: {args.outfile}")

    # Quick stats: agreement vs original batched single_response, and accuracy if gt_answer present
    n_total = 0
    n_agree = 0
    n_correct_iso = 0
    n_correct_batched = 0
    n_with_gt = 0
    for si, sample in enumerate(data[: args.num_samples]):
        for turn in sample.get("conversations", []):
            for qa in turn.get("dsg_qa", []):
                if "single_response_isolated" not in qa:
                    continue
                n_total += 1
                iso = qa["single_response_isolated"]
                bat = qa.get("single_response", "")
                if iso == bat:
                    n_agree += 1
                gt = qa.get("gt_answer") or qa.get("answer")
                if gt:
                    n_with_gt += 1
                    g = "Yes" if str(gt).strip().lower().startswith("y") else "No"
                    if iso == g:
                        n_correct_iso += 1
                    if bat == g:
                        n_correct_batched += 1
    print(f"\n=== Comparison over {n_total} POPE Qs in first {args.num_samples} samples ===")
    print(f"agreement (isolated vs batched): {n_agree}/{n_total} = {100*n_agree/max(n_total,1):.2f}%")
    if n_with_gt:
        print(f"accuracy (isolated, single-call): {n_correct_iso}/{n_with_gt} = {100*n_correct_iso/n_with_gt:.2f}%")
        print(f"accuracy (batched per-turn):     {n_correct_batched}/{n_with_gt} = {100*n_correct_batched/n_with_gt:.2f}%")


if __name__ == "__main__":
    main()
