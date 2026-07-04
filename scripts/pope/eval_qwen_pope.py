#!/usr/bin/env python3
"""
Re-run Qwen2.5-VL-7B-Instruct on each dsg_qa question with the corresponding image (from url),
then update single_response_raw, single_response, and single_is_correct (parsing and ref logic from DSG_v4).
"""

import json
import os
import re
import sys
import argparse
import multiprocessing as mp
from typing import Dict, Any, List, Tuple, Optional

# Add project root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import tqdm

# Paths
INPUT_JSON = "/raid/william/project/context-eval-mllm/work_dirs/vg/final_run_v18_gpt4o_completed/Qwen2.5-VL-7B-Instruct_with_both_answers.json"
DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"


def normalize_answer(a):
    if not a:
        return "unknown"
    a = a.lower().strip()
    if "unknown" in a or "unclear" in a or "n/a" in a:
        return "unknown"
    if "yes" in a and "no" not in a:
        return "yes"
    if "no" in a:
        return "no"
    return "unknown"


def normalize_qa_answer_to_yes_no(text):
    """Normalize qa['answer'] to Yes/No/Unknown (from DSG_v4)."""
    if not text or not str(text).strip():
        return "Unknown"
    t = str(text).lower().strip()
    if "yes" in t and "no" not in t:
        return "Yes"
    if "no" in t:
        return "No"
    return "Unknown"


def _effective_ref_verify_only(qa):
    """Reference for correctness: gt_answer if not Unknown, else qa['answer'] (from DSG_v4)."""
    gt = qa.get("gt_answer")
    if gt is not None and gt != "Unknown" and normalize_answer(gt) != "unknown":
        return gt
    return normalize_qa_answer_to_yes_no(qa.get("answer", ""))


def compute_accuracy(extracted, gt):
    en = normalize_answer(extracted)
    gn = normalize_answer(gt)
    if gn == "unknown":
        return {"is_correct": False, "skip_reason": "gt_unknown"}
    if en == "unknown":
        return {"is_correct": None, "skip_reason": "extracted_unknown"}
    return {"is_correct": (en == gn), "skip_reason": None}


def parse_yesno_list(raw: str, n_expected: int) -> list:
    """Parse VLM raw output into list of Yes/No/Unknown (from DSG_v4)."""
    if not raw or not raw.strip():
        return ["Unknown"] * n_expected
    text = raw.strip()
    out = []
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for ln in lines:
        ln_clean = re.sub(r"^\s*\d+[\.\)]\s*", "", ln.lower(), flags=re.I).strip()
        if ln_clean in ("yes", "y"):
            out.append("Yes")
        elif ln_clean in ("no", "n"):
            out.append("No")
        elif "yes" in ln_clean and "no" not in ln_clean:
            out.append("Yes")
        elif "no" in ln_clean:
            out.append("No")
        else:
            out.append("Unknown")
    if len(out) < n_expected:
        tokens = re.findall(r"\b(yes|no|y|n)\b", text, re.I)
        extracted = []
        for t in tokens:
            if t.lower() in ("yes", "y"):
                extracted.append("Yes")
            else:
                extracted.append("No")
        if len(extracted) > len(out):
            out = extracted
    if len(out) < n_expected and len(out) > 0:
        last = out[-1] if out[-1] in ("Yes", "No") else "Unknown"
        while len(out) < n_expected:
            out.append(last)
    while len(out) < n_expected:
        out.append("Unknown")
    return out[:n_expected]


def _shard_indices(n: int, shard_id: int, num_shards: int) -> List[int]:
    return [i for i in range(n) if (i % num_shards) == shard_id]


def _worker_run(
    shard_id: int,
    num_shards: int,
    gpu_id: str,
    model_path: str,
    input_path: str,
    first_n: Optional[int],
    out_dir: str,
) -> str:
    """
    One GPU worker:
    - restricts visibility to a single GPU
    - loads the model once
    - processes its shard of sample indices
    - writes results as a dict: {sample_index: updated_sample}
    Returns the path to the shard output JSON.
    """
    # IMPORTANT: set before importing torch/transformers
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import inside worker after CUDA_VISIBLE_DEVICES is set
    from infer.loader import load_model  # noqa: WPS433

    with open(input_path, "r") as f:
        data = json.load(f)
    if first_n is not None:
        data = data[: first_n]

    loader_args = type("Args", (), {
        "model_path": model_path,
        "model_base": None,
        "use_conversation": False,
    })()
    vlm_eval_fn = load_model(loader_args)

    indices = _shard_indices(len(data), shard_id, num_shards)
    out_map: Dict[int, Any] = {}

    for idx in indices:
        record = data[idx]
        url = record.get("url")
        if not url:
            out_map[idx] = record
            continue

        for conv in record.get("conversations", []):
            for qa in conv.get("dsg_qa", []):
                question = qa.get("question", "")
                if not question:
                    continue
                try:
                    # loader returns a partial with processor/model bound; pass remaining by name
                    raw = vlm_eval_fn(image_file=url, query=question)
                    if isinstance(raw, tuple):
                        raw = raw[0] if len(raw) > 0 else ""
                    raw = (raw or "").strip()
                except Exception:
                    raw = ""
                qa["single_response_raw"] = raw
                parsed = parse_yesno_list(raw, 1)
                qa["single_response"] = parsed[0] if parsed else "Unknown"
                ref = _effective_ref_verify_only(qa)
                acc = compute_accuracy(qa["single_response"], ref)
                qa["single_is_correct"] = acc["is_correct"]

        out_map[idx] = record

    os.makedirs(out_dir, exist_ok=True)
    shard_path = os.path.join(out_dir, f"qwen_pope_shard_{shard_id}.json")
    with open(shard_path, "w") as f:
        json.dump(out_map, f)
    return shard_path


def main():
    parser = argparse.ArgumentParser(description="Eval Qwen2.5-VL on pope both_answers JSON; update single_response_raw, single_response, single_is_correct")
    parser.add_argument("--input", type=str, default=INPUT_JSON, help="Input JSON path")
    parser.add_argument("--out", type=str, default=None, help="Output JSON (default: overwrite input)")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Qwen2.5-VL model path (must contain Qwen2.5-VL)")
    parser.add_argument("--first_n", type=int, default=None, help="Process only first N samples (for debugging)")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5", help="Comma-separated GPU ids to use (default: 0-5)")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: len(--gpus))")
    parser.add_argument("--tmp_dir", type=str, default=None, help="Temp dir for shard outputs (default: <out_dir>/.tmp_qwen_pope)")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    with open(args.input, "r") as f:
        data = json.load(f)

    if args.first_n is not None:
        data = data[: args.first_n]
        print(f"Limiting to first {args.first_n} samples")
        out_path = args.out or (args.input.replace(".json", "") + "_first%d.json" % args.first_n)
    else:
        out_path = args.out or args.input

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    if not gpu_ids:
        raise SystemExit("No GPUs specified in --gpus")
    num_workers = args.workers or len(gpu_ids)
    num_workers = min(num_workers, len(gpu_ids))

    out_dir = os.path.dirname(out_path) or "."
    tmp_dir = args.tmp_dir or os.path.join(out_dir, ".tmp_qwen_pope")

    # Multiprocessing: one process per GPU, each loads model once
    ctx = mp.get_context("spawn")
    shard_paths: List[str] = []
    print(f"Using {num_workers} worker(s) on GPUs: {gpu_ids[:num_workers]}")
    with ctx.Pool(processes=num_workers) as pool:
        jobs = []
        for shard_id in range(num_workers):
            gpu_id = gpu_ids[shard_id]
            jobs.append(pool.apply_async(
                _worker_run,
                (shard_id, num_workers, gpu_id, args.model_path, args.input, args.first_n, tmp_dir),
            ))
        for j in tqdm.tqdm(jobs, desc="Workers"):
            shard_paths.append(j.get())

    # Merge shard outputs back into `data` (preserve order)
    merged = data
    total_updates = 0
    for sp in shard_paths:
        with open(sp, "r") as f:
            out_map = json.load(f)
        for k, rec in out_map.items():
            idx = int(k)
            merged[idx] = rec
            total_updates += 1

    print(f"Merged {total_updates} sample updates from {len(shard_paths)} shard file(s)")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
