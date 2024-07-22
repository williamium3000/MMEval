from dyna.data import load_coco2017
from dyna.utils import call_chatgpt
from dyna.prompt import CONVERSATION_PERSONA_PROMPT, PERSONA_PROMPT, CONVERSATION_PERSONA_SELECTION_PROMPT
from infer.infer_llava import load_model, eval_model
import os
import argparse
import json
import tqdm
import re


def parse_json(text):
    pattern = r"```json(.*)```"
    match = re.search(pattern, text, re.DOTALL)
    json_text = match.group(1) if match else text
    return json.loads(json_text)


def generate_persona(case, persona_type):
    if persona_type == 'dynamic':
        prompt = PERSONA_PROMPT.format(case)
    elif persona_type == 'default':
        prompt = CONVERSATION_PERSONA_SELECTION_PROMPT.format(case)

    conversations = [
        {"role": "system", "content": "You are a helpful and precise prompt generator for vision-language model."},
        {"role": "user", "content": prompt}
    ]
    while True:
        try:
            message = call_chatgpt(conversations)
            persona = parse_json(message)
            break
        except Exception as e:
            print(e)
            continue

    return persona


def dyna_conv(persona, case):
    prompt = CONVERSATION_PERSONA_PROMPT.format(persona["persona"], persona["objective"],persona["question"], case)

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
    parser.add_argument('--persona_type', type=str, default="default", choices=["dynamic", "default"])
    parser.add_argument('--outdir', type=str, default="output/coco2017")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # need to figure out how to eval on different models
    model_name, tokenizer, model, image_processor, context_len = load_model(args.model_path, args.model_base)
    model_path = args.model_path
    samples = load_coco2017(args.debug)

    output_path = os.path.join(args.outdir, f"conversation_{args.persona_type}_persona.json")

    print("starting conversation with model...")
    for sample in tqdm.tqdm(samples):
        persona_list = generate_persona(sample, args.persona_type)

        conversation_list = []
        for persona in persona_list:
            conv = dyna_conv(persona, sample)
            conv_dic = {"persona": persona, "conversation": conv}
            conversation_list.append(conv_dic)

        sample["conversation_list"] = conversation_list

    with open(output_path, "w") as f:
        json.dump(samples, f, indent=4)
