from dyna.data import load_coco2017, format_case_coco
from dyna.utils import call_chatgpt_json
from dyna.promptv2 import CONV_COVERAGE_PROMPT_CERTAINTY, CONV_COVERAGE_PROMPT, CONV_COVERAGE_PROMPT_CERTAINTY_WITH_ANSWER
from infer.infer_llava import load_model, eval_model
import os
import argparse
import json
import tqdm
import copy

def dyna_conv(case):
    prompt = CONV_COVERAGE_PROMPT_CERTAINTY_WITH_ANSWER.format(format_case_coco(case))
    conversations = [
                    {"role": "system", "content": "You are a helpful AI visual assistant that can analyze a single image and capable of having a conversation with a human."},
                    {"role": "user", "content": prompt}
    ]
    
    to_save = []
    r = 0
    while True:
        message = call_chatgpt_json(conversations)
        message_json = json.loads(message)
        message_evaluator, gt_answer = message_json["prompt"], message_json["response"]
        
        print(message)
        
        if "END" in message_evaluator:
            break
        
        conversations.append({"role": "assistant", "content": message})
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
        conversations.append({"role": "user", "content": output})
        r += 1
        to_save.append(
            {"round_id": r, "prompt": message_evaluator, "response":output, "gt": gt_answer}
        )
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
