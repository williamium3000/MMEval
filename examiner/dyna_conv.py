from utils.utils import load_data
from utils.vg import format_case_vg
from utils.coco import format_case_coco
from utils.llm import LLMChat, parse_json
from examiner import prompt as PROMPT
from infer.loader import load_model
import os
import argparse
import json
import tqdm
import copy
import traceback


CONV_PROMPT = \
"""
Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""

def dyna_conv(args, case, llm_chat, eval_func):
    sys_prompt = PROMPT.__dict__[args.p_mode]
    image_info = format_case_vg(case) if args.dataset == "vg" else format_case_coco(case)
    loaded_icls = []
    if args.icls is not None:
        loaded_icls = json.load(open(args.icls))
    
    ICLs = []
    for icl in loaded_icls:
        image_info = format_case_vg(icl["image_info"]) if args.dataset == "vg" else format_case_coco(icl["image_info"])
        firstp = CONV_PROMPT.format(image_info)
        ICLs.append({"role": "user", "content": firstp})
        ICLs.extend(icl["conversations"])
        
    conversations = [
                    {"role": "system", "content": sys_prompt},
                    *ICLs,
                    {"role": "user", "content": CONV_PROMPT.format(image_info)}
    ]
    to_save = []
    r = 0
    while True:
        message_evaluator = llm_chat.chat(conversations, None)
        
        if "END" in message_evaluator:
            break
        
        conversations.append({"role": "assistant", "content": message_evaluator})
        image_file = case["image"]
        output = eval_func(image_file=image_file, query=message_evaluator)
        output = output.lower()
        conversations.append({"role": "user", "content": output})
        # print(f"examiner: {message_evaluator}")
        # print(f"vlm model: {output}")
        r += 1
        to_save.append(
            {"round_id": r, "prompt": message_evaluator, "response":output}
        )
    return to_save


def load_cache(cache_file):
    """Load existing results from cache file if it exists.
    
    Returns:
        list: List of cached conversation results
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            print(f"Loaded {len(cached_data)} cached conversation results from {cache_file}")
            return cached_data
        except json.JSONDecodeError as e:
            print(f"Warning: Cache file {cache_file} is corrupted (JSON decode error: {e})")
            print("Starting fresh...")
            return []
        except Exception as e:
            print(f"Warning: Error loading cache file {cache_file}: {e}")
            return []
    else:
        return []

def save_cache(cache_file, data):
    """Save results to cache file using atomic write to prevent corruption."""
    try:
        # Use atomic write: write to temp file first, then rename
        # This ensures the original file is only replaced if write succeeds
        temp_file = cache_file + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        
        # Atomic rename - only succeeds if temp file was written completely
        os.replace(temp_file, cache_file)
        print(f"Saved {len(data)} results to cache file {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save to cache file {cache_file}: {e}")
        traceback.print_exc()
        # Clean up temp file if it exists
        temp_file = cache_file + '.tmp'
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

def get_sample_id(sample):
    """Generate a unique identifier for a sample based on its content."""
    # Use image_id as the primary identifier
    return sample.get("image_id", "unknown")

def is_sample_cached(sample_id, cached_data):
    """Check if a sample has already been processed.
    
    Args:
        sample_id: The image_id to look for
        cached_data: List of all cached conversation results
    
    Returns:
        bool: True if sample is already cached
    """
    for item in cached_data:
        if item.get("image_id") == sample_id:
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument("--p_mode", type=str)
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--icls', type=str, default=None)
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--cache_file', type=str, default=None, 
                       help='Cache file to store/load intermediate results for resuming')
    args = parser.parse_args()

    # Set default cache file if not provided
    if args.cache_file is None:
        args.cache_file = args.outfile.replace('.json', '_cache.json')

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    # need to figure out how to eval on different models
    eval_func = load_model(args)
    samples = load_data(args)
    
    llm_chat = LLMChat(model_name="gpt-4o")
    
    # Initialize cache and resume functionality
    to_save = []
    cached_data = []
    
    # Load existing cache if provided
    if args.cache_file:
        cached_data = load_cache(args.cache_file)
        to_save.extend(cached_data)
        print(f"Loaded {len(cached_data)} cached samples")
    
    print("starting conversation with model...")
    total_samples = len(samples)
    completed_samples = len(cached_data)
    
    for i, sample in enumerate(tqdm.tqdm(samples, desc="Processing samples")):
        sample_id = get_sample_id(sample)
        
        # Check if sample is already cached (1 per image rule)
        if is_sample_cached(sample_id, cached_data):
            print(f"Skipping sample {i+1}/{total_samples} (already processed): {sample_id}")
            continue
        
        print(f"Processing sample {i+1}/{total_samples}: {sample_id}")
        
        try:
            conv = dyna_conv(args, sample, llm_chat, eval_func)
            sample_to_save = copy.deepcopy(sample)
            sample_to_save["conversations"] = conv
            del sample_to_save["image"]
            to_save.append(sample_to_save)
            completed_samples += 1
            
            # Save cache incrementally if cache file is provided
            if args.cache_file:
                save_cache(args.cache_file, to_save)
                print(f"Progress: {completed_samples}/{total_samples} samples completed")
                
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            print("Continuing with next sample...")
            traceback.print_exc()
            continue
    
    # Final save to output file using atomic write
    temp_file = args.outfile + '.tmp'
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(to_save, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        os.replace(temp_file, args.outfile)  # Atomic rename
    except Exception as e:
        print(f"Error saving output file: {e}")
        traceback.print_exc()
        # Clean up temp file if it exists
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        raise
    
    print(f"Completed processing {completed_samples}/{total_samples} samples")
    print(f"Results saved to {args.outfile}")
    if args.cache_file:
        print(f"Cache saved to {args.cache_file}")
