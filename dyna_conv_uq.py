from dyna.data import load_coco2017, format_case_coco
from dyna.utils import call_chatgpt
from dyna.prompt import CONVERSATION_UQ_PROMPT, CONVERSATION_ADVERSARIAL_PROMPT, CONVERSATION_PROMPT
from infer.infer_llava import load_model, eval_model
import os
import argparse
import json
import tqdm
import nltk

nltk.download('wordnet')
from nltk.corpus import wordnet as wn


def dyna_conv(case, question_type, include_image=False):
    formatted_case = format_case_coco(case)
    if question_type == 're':
        prompt = CONVERSATION_PROMPT.format(formatted_case)
    elif question_type == 'uq':
        prompt = CONVERSATION_UQ_PROMPT.format(formatted_case)
    elif question_type == 'ad':
        instance_list = []
        for instance in sample["instances"]:
            object_name = instance["category"]
            result_list = []
            for word_list in wn.synonyms(object_name):
                result_list += word_list
            instance["relevant_words"] = result_list
            instance_list.append(instance)
        case["instances"] = instance_list
        prompt = CONVERSATION_ADVERSARIAL_PROMPT.format(case)

    if include_image:
        conversations = [
            {"role": "system",
             "content": "You are a helpful AI visual assistant that can analyze a single image and capable of having "
                        "a conversation with a human."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": case["image_url"],
                            "detail": "auto"
                        },
                    },
                ],
            }
        ]
    else:
        conversations = [
            {"role": "system", "content": "You are a helpful AI visual assistant that can analyze a single image and capable of having "
                        "a conversation with a human."},
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
    parser.add_argument('--question_type', type=str, default=None, choices=['re', 'uq', 'ad'])
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--include_image', action="store_true")
    parser.add_argument('--outfile', type=str, default="output/revised_coco2017/conversation.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    # need to figure out how to eval on different models
    model_name, tokenizer, model, image_processor, context_len = load_model(args.model_path, args.model_base)
    model_path = args.model_path
    samples = load_coco2017(args.debug)

    print("starting conversation with model...")
    for sample in tqdm.tqdm(samples):
        conv = dyna_conv(sample, args.question_type, args.include_image)
        sample["conversation"] = conv

    with open(args.outfile, "w") as f:
        json.dump(samples, f, indent=4)
