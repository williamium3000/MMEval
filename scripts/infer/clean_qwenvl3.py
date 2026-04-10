#!/usr/bin/env python3
"""
Detect consecutive substring repetition in conversation responses.
Reads a JSON like work_dirs/vg/ablation_baseline2/Qwen3-VL-8B-Instruct.json,
checks each item's conversations[].response for any substring (length > 3)
repeated more than 5 times consecutively (e.g. generation loop).
Reports image_ids that have such repetition and count (X out of Y).

Usage:
  python scripts/infer/clean_qwenvl3.py [path/to/file.json]
  bash scripts/infer/clean_qwenvl3.py   # if run from repo root (python inferred)
"""

import json
import argparse
import os
import sys

MIN_SUBSTRING_LEN = 4   # substring length > 3
MIN_REPEATS = 6         # repeated > 5 times means 6+ consecutive copies
MAX_PERIOD = 500        # max substring length to check (avoid O(n²) blowup)


def has_consecutive_repetition(text, min_len=MIN_SUBSTRING_LEN, min_repeats=MIN_REPEATS, max_period=MAX_PERIOD):
    """
    Return True if text contains a substring s (min_len <= len(s) <= max_period)
    repeated min_repeats or more times consecutively.
    """
    if not text or len(text) < min_len * min_repeats:
        return False
    n = len(text)
    max_l = min(max_period, n // min_repeats)
    for L in range(min_len, max_l + 1):
        for start in range(0, n - L * min_repeats + 1):
            block = text[start : start + L]
            if text[start : start + L * min_repeats] == block * min_repeats:
                return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Detect consecutive substring repetition in conversation responses (report image_id and X out of Y)"
    )
    default_path = "work_dirs/vg/ablation_baseline2/Qwen3-VL-8B-Instruct.json"
    parser.add_argument(
        "input_json",
        nargs="?",
        default=default_path,
        help="Path to JSON file (list of items with image_id and conversations[].response)",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=MIN_SUBSTRING_LEN,
        help=f"Minimum substring length (default: {MIN_SUBSTRING_LEN})",
    )
    parser.add_argument(
        "--min-repeats",
        type=int,
        default=MIN_REPEATS,
        help=f"Minimum consecutive repeats to flag (default: {MIN_REPEATS})",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove items with repetition from the JSON and overwrite the file",
    )
    args = parser.parse_args()

    path = args.input_json
    if not os.path.isabs(path) and not os.path.exists(path):
        # Try relative to script dir / repo root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(script_dir))
        alt = os.path.join(repo_root, path)
        if os.path.exists(alt):
            path = alt
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Expected JSON array of items.", file=sys.stderr)
        sys.exit(1)

    total = len(data)
    affected_ids = []

    for item in data:
        image_id = item.get("image_id", "unknown")
        conversations = item.get("conversations") or []
        for conv in conversations:
            response = conv.get("response") or ""
            if has_consecutive_repetition(
                response,
                min_len=args.min_len,
                min_repeats=args.min_repeats,
            ):
                affected_ids.append(image_id)
                break  # one hit per item is enough

    count = len(affected_ids)
    affected_set = set(affected_ids)
    print(f"Consecutive repetition (substring len > {args.min_len - 1}, repeated > {args.min_repeats - 1} times):")
    print(f"  Affected image_ids: {count} out of {total}")
    if affected_ids:
        print(f"  image_ids: {affected_ids}")
    else:
        print("  image_ids: (none)")

    if args.remove and affected_set:
        cleaned = [item for item in data if item.get("image_id") not in affected_set]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=4)
        print(f"  Removed {len(affected_set)} item(s). File now has {len(cleaned)} items.")

    return 0 if count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
