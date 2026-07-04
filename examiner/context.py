import re

from PIL import Image

from utils.utils import load_data
from utils.vg import format_case_vg
from utils.coco import format_case_coco
from utils.llm import LLMChat
from examiner import prompt as PROMPT
from infer.loader import load_model
import os
import argparse
import json
import tqdm
import copy

CONTEXT_PROMPT = \
"""
Image information:
{}

Please generate two contexts. You MUST only respond in the format as described above. DO NOT RESPOND WITH ANYTHING ELSE.
"""

CONV_PROMPT = \
"""
The given context is
background: {}

objective: {}

Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""


def parse_json(text): 
    pattern = r"```json(.*)```"
    match = re.search(pattern, text, re.DOTALL)
    json_text = match.group(1) if match else text
    return json.loads(json_text)


def generate_context(case, llm_chat):
    image_info = format_case_vg(case) if args.dataset == "vg" else format_case_coco(case)
    conversations = [
        {"role": "system", "content": PROMPT.CONTEXT_PROMPT.strip()},
        {"role": "user", "content": CONTEXT_PROMPT.format(image_info).strip()}
    ]

    message = llm_chat.chat(conversations, None)
    # print(message)
    contexts = parse_json(message)

    return contexts


def dyna_conv(args, context, case, llm_chat, eval_func):
    image_info = format_case_vg(case) if args.dataset == "vg" else format_case_coco(case)
    loaded_icls = []
    if args.icls is not None:
        loaded_icls = json.load(open(args.icls))
    
    ICLs = []
    for icl in loaded_icls:
        image_info = format_case_vg(icl["image_info"]) if args.dataset == "vg" else format_case_coco(icl["image_info"])
        firstp = CONV_PROMPT.format(icl["context"]["background"], icl["context"]["goal"], image_info)
        ICLs.append({"role": "user", "content": firstp})
        ICLs.extend(icl["conversations"])
        
    conversations = [
                    {"role": "system", "content": PROMPT.__dict__[args.p_mode]},
                    *ICLs,
                    {"role": "user", "content": CONV_PROMPT.format(context["background"], context["goal"], image_info)}
    ]
    
    to_save = []
    r = 0
    while True:
        message_evaluator = llm_chat.chat(conversations, None)
        
        if "end" in message_evaluator.lower():
            break
        
        conversations.append({"role": "assistant", "content": message_evaluator})
        image_file = case["image"]
        # image_file = Image.open(case["image"]).convert("RGB")
        output = eval_func(image_file=image_file, query=message_evaluator)
        output = output.lower()
        conversations.append({"role": "user", "content": output})
        print(f"examiner: {message_evaluator}")
        print(f"vlm model: {output}")
        r += 1
        to_save.append(
            {"round_id": r, "prompt": message_evaluator, "response":output}
        )
    return to_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument("--p_mode", type=str, default="certainty")
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--icls', type=str, default=None)
    parser.add_argument('--outfile', type=str)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    # need to figure out how to eval on different models
    eval_func = load_model(args)
    samples = load_data(args)
    # with open('output/vg_samples.json', 'r') as json_file:
    #     samples = json.load(json_file)
    
    llm_chat = LLMChat(model_name="gpt-4o")
    
    to_save = []
    print("starting conversation with model...")
    for sample in tqdm.tqdm(samples):
        contexts = generate_context(sample, llm_chat)
        for context in contexts:
            conv = dyna_conv(args, context, sample, llm_chat, eval_func)
            sample_to_save = copy.deepcopy(sample)
            del sample_to_save["image"]
            sample_to_save["conversations"] = conv
            sample_to_save["context"] = context
            to_save.append(sample_to_save)
        
    
    with open(args.outfile, "w") as f:
        json.dump(to_save, f, indent=4)
