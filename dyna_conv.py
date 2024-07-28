from dyna.data import load_coco2017, format_case_coco
from dyna.utils import call_chatgpt
from dyna.prompt import CONVERSATION_PROMPT, CONVERSATION_PROMPT_GROUND_TRUTH
from dyna.promptv2 import CONV_COVERAGE_PROMPT_CERTAINTY, CONV_COVERAGE_PROMPT
from infer.infer_llava import load_model, eval_model
import os
import argparse
import json
import tqdm
import copy

def dyna_conv(case, with_ground_truth):
    # if with_ground_truth:
    #     prompt = CONVERSATION_PROMPT_GROUND_TRUTH.format(case)
    # else:
    #     prompt = CONVERSATION_PROMPT.format(case)
    #
    prompt = CONV_COVERAGE_PROMPT_CERTAINTY.format(format_case_coco(case))
    conversations = [
                    {"role": "system", "content": "You are a helpful AI visual assistant that can analyze a single image and capable of having a conversation with a human."},
                    {"role": "user", "content": prompt}
    ]
    #
    # conversations = [
    #         {"role": "system", "content": "You are a helpful conversation-based evaluator."},
    #         {"role": "user", "content": prompt}
    # ]
    
    to_save = []
    ground_truth_conversation = []

    while True:
        message_evaluator = call_chatgpt(conversations)
        if "END" in message_evaluator:
            break

        if with_ground_truth:
            dictionary = json.loads(message_evaluator.strip())
            message_evaluator = dictionary["question"]

            ground_truth_conversation += [{"question": message_evaluator}, {"ground_truth": dictionary["ground_truth"]}]

        to_save.append({"role": "evaluator", "content": message_evaluator})
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
        output = output.lower()
        conversations.append({"role": "user", "content": output})
        to_save.append({"role": "evaluatee", "content": output})
    return to_save, ground_truth_conversation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--ground_truth', action="store_true")
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
        conv, ground_truth_conversation = dyna_conv(sample, args.ground_truth)
        sample["conversation"] = conv
        if args.ground_truth:
            sample["ground_truth"] = ground_truth_conversation
    
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=4)
