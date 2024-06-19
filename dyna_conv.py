from dyna.data import load_coco2017
from dyna.utils import call_chatgpt
from dyna.prompt import CONVERSATION_PROMPT
from  infer.infer_llava import load_model, eval_model
import os
import argparse
import json
import tqdm

def dyna_conv(case):
    prompt = CONVERSATION_PROMPT.format(case)
    conversations = [
                    {"role": "system", "content": "You are a helpful conversation-based evaluator."},
                    {"role": "user", "content": prompt}
    ]
    
    to_save = []
    while True:
        message_evaluator = call_chatgpt(conversations)
        to_save.append({"role": "evaluator", "content": message_evaluator})
        if "END" in message_evaluator:
            break
        
        conversations.append({"role": "assistant", "content": message_evaluator})
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
        output = output.strip().replace(".", '').lower()
        conversations.append({"role": "user", "content": output})
        to_save.append({"role": "evaluatee", "content": output})
    return to_save

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--outdir', type=str, default="output/coco2017")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # need to figure out how to eval on different models
    model_name, tokenizer, model, image_processor, context_len = load_model(args.model_path, args.model_base)
    model_path = args.model_path
    samples = load_coco2017(args.debug)
    
    output_path = os.path.join(args.outdir, "conversation.json")
    
    print("starting conversation with model...")
    for sample in tqdm.tqdm(samples):
        conv = dyna_conv(sample)
        sample["conversation"] = conv
    
    
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=4)
