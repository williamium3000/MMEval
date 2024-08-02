from utils.coco import load_coco2017, format_case_coco
from utils.utils import call_chatgpt, call_chatgpt_json
from infer.infer_llava import load_model, eval_model
import os
import argparse
import json
import tqdm
import copy

def dyna_conv(case):
    message_evaluator = "Please provide a detailed description."
    image_file = os.path.join("data/coco/val2017", case["file_name"])
    output = eval_model(model_name, tokenizer, model, image_processor, context_len, type('Args', (), {
                            "model_path": model_path,
                            "model_base": None,
                            "model_name": model_name,
                            "query": message_evaluator,
                            "conv_mode": None,
                            "image_file": image_file,
                            "sep": ",",
                            "load_in_8bit": False,
                            "load_in_4bit": False,
                            "temperature": 0.0,  # set as 0.0 for reproceduce
                            "top_p": None,
                            "num_beams": 1,
                            "max_new_tokens": 512
                        })())
    output = output.lower()
    to_save = [
        {"round_id": 0, "prompt": "Please provide a brief description.", "response": output}
    ]
    
    return to_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--outfile', type=str)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    # need to figure out how to eval on different models
    model_name, tokenizer, model, image_processor, context_len = load_model(args.model_path, args.model_base)
    model_path = args.model_path
    samples = load_coco2017(args.debug)
    
    print("starting conversation with model...")
    for sample in tqdm.tqdm(samples):
        conv = dyna_conv(sample)
        sample["conversations"] = conv
    
    
    with open(args.outfile, "w") as f:
        json.dump(samples, f, indent=4)
