from utils.utils import load_data, load_image
from infer.loader import load_model
import os
import argparse
import json
import tqdm
import copy
import traceback
import urllib.request
from PIL import Image


CONTEXT_PROMPT_PREFIX = "Considering this context: {context}\n"
CAPTION_QUERY = "Please provide a detailed description."

VG_IMAGE_CACHE_DIR = os.path.join("work_dirs", "vg_image_cache")


def _build_image_url_map():
    """Map image_id -> canonical Visual Genome URL from utils.vg._image_data_raw."""
    from utils.vg import _image_data_raw
    return {r["image_id"]: r["url"] for r in _image_data_raw}


def _resolve_image_path(image_id, url_map):
    """Download VG image to local cache if needed, return local path."""
    os.makedirs(VG_IMAGE_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(VG_IMAGE_CACHE_DIR, f"{image_id}.jpg")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path
    url = url_map.get(image_id)
    if not url:
        raise ValueError(f"No URL found for image_id={image_id}")
    urllib.request.urlretrieve(url, cache_path)
    return cache_path


def get_image_for_sample(case, url_map):
    """Return a PIL.Image for the sample. Prefers an in-memory image already
    attached, else loads from VG URL via local cache."""
    img = case.get("image")
    if img is not None:
        return img
    image_id = case["image_id"]
    path = _resolve_image_path(image_id, url_map)
    return Image.open(path).convert("RGB")


def load_contexts(context_file):
    """Build {image_id: context_dict} from a v18 dyna run JSON, taking the first
    occurrence per image_id."""
    with open(context_file, "r") as f:
        data = json.load(f)
    contexts = {}
    for entry in data:
        image_id = entry.get("image_id")
        ctx = entry.get("context")
        if image_id is None or not isinstance(ctx, dict):
            continue
        if image_id in contexts:
            continue
        contexts[image_id] = ctx
    print(f"Loaded {len(contexts)} contexts from {context_file}")
    return contexts


def format_context(ctx):
    background = (ctx.get("background") or "").strip()
    goal = (ctx.get("goal") or "").strip()
    parts = [p for p in (background, goal) if p]
    return " ".join(parts)


def dyna_conv(case, eval_func, contexts, url_map):
    image_id = case.get("image_id")
    ctx = contexts.get(image_id)
    if ctx is None:
        raise ValueError(f"No context found for image_id={image_id}")
    context_text = format_context(ctx)
    if not context_text:
        raise ValueError(f"Empty context for image_id={image_id}")
    image_file = get_image_for_sample(case, url_map)
    query = CONTEXT_PROMPT_PREFIX.format(context=context_text) + CAPTION_QUERY
    output = eval_func(image_file=image_file, query=query)
    output = output.lower()
    to_save = [
        {"round_id": 0, "prompt": query, "response": output}
    ]
    return to_save


def load_cache(cache_file):
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
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--context_file', type=str, required=True,
                        help="Path to v18 dyna run JSON containing per-image_id 'context' fields.")
    parser.add_argument('--cache_file', type=str, default=None,
                        help="Cache file used for resuming. Defaults to <outfile>_cache.json")
    args = parser.parse_args()

    if args.cache_file is None:
        args.cache_file = args.outfile.replace(".json", "_cache.json")

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    contexts = load_contexts(args.context_file)

    eval_func = load_model(args)
    samples = load_data(args)
    url_map = _build_image_url_map()

    print("starting contextualized captioning with model...")

    to_save = []
    cached_data, cache_index = load_cache(args.cache_file) if args.cache_file else ([], {})
    if cached_data:
        to_save.extend(cached_data)

    completed = len(cached_data)
    skipped_missing_context = 0
    for sample in tqdm.tqdm(samples, desc="Processing samples"):
        image_id = sample.get("image_id")
        if image_id is not None and cache_index.get(image_id, False):
            continue
        if image_id not in contexts:
            skipped_missing_context += 1
            continue
        try:
            conv = dyna_conv(sample, eval_func, contexts, url_map)
            sample_to_save = copy.deepcopy(sample)
            sample_to_save["conversations"] = conv
            sample_to_save["context"] = contexts[image_id]
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
    if skipped_missing_context:
        print(f"Skipped {skipped_missing_context} samples with no matching context.")
    if args.cache_file:
        print(f"Cache saved to {args.cache_file}")
