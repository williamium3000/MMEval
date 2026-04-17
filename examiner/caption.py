from utils.utils import load_data
from infer.loader import load_model
import os
import argparse
import json
import tqdm
import copy
import traceback

def dyna_conv(case, eval_func):
    message_evaluator = "Please provide a detailed description."
    image_file = case["image"]
    output = eval_func(image_file=image_file, query=message_evaluator)
    output = output.lower()
    to_save = [
        {"round_id": 0, "prompt": "Please provide a brief description.", "response": output}
    ]
    
    return to_save


def load_cache(cache_file):
    """Load existing results from cache file if it exists."""
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached_data = json.load(f)
        print(f"Loaded {len(cached_data)} cached caption results from {cache_file}")
        cache_index = {}
        for item in cached_data:
            if "image_id" in item:
                cache_index[item["image_id"]] = True
        return cached_data, cache_index
    return [], {}


def save_cache(cache_file, data):
    """Save cache incrementally."""
    try:
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved {len(data)} results to cache file {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save to cache file {cache_file}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument(
        "--cache_file",
        type=str,
        default=None,
        help="Cache file used for resuming. Defaults to <outfile>_cache.json",
    )
    args = parser.parse_args()

    if args.cache_file is None:
        args.cache_file = args.outfile.replace(".json", "_cache.json")

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    # need to figure out how to eval on different models
    eval_func = load_model(args)
    samples = load_data(args)
    
    print("starting conversation with model...")

    # Resume support: load cached results and skip completed image_ids
    to_save = []
    cached_data, cache_index = load_cache(args.cache_file) if args.cache_file else ([], {})
    if cached_data:
        to_save.extend(cached_data)

    completed = len(cached_data)
    for sample in tqdm.tqdm(samples, desc="Processing samples"):
        image_id = sample.get("image_id")
        if image_id is not None and cache_index.get(image_id, False):
            continue
        try:
            conv = dyna_conv(sample, eval_func)
            sample_to_save = copy.deepcopy(sample)
            sample_to_save["conversations"] = conv
            if "image" in sample_to_save:
                del sample_to_save["image"]
            to_save.append(sample_to_save)
            completed += 1
            if args.cache_file:
                save_cache(args.cache_file, to_save)
        except Exception as e:
            print(f"Error processing sample {image_id}: {e}")
            traceback.print_exc()
            continue

    with open(args.outfile, "w") as f:
        json.dump(to_save, f, indent=4)
    print(f"Completed captions: {completed}/{len(samples)}")
    if args.cache_file:
        print(f"Cache saved to {args.cache_file}")
