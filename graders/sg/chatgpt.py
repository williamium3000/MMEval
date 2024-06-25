from dyna.utils import call_chatgpt
from dyna.prompt import LSG_PROMPT
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

def parse_scene_graph(sample):
    conversation = sample["conversation"]
    conversation_to_be_evaluated = []
    for turn in conversation:
        if turn["role"] == "evaluatee":
            conversation_to_be_evaluated.append(turn["content"])
    conversation = " ".join(conversation_to_be_evaluated)
    
    prompt = LSG_PROMPT.format(conversation)
    conversations = [
                    {"role": "system", "content": "You are a helpful scene graph parser that can parse scene graphs from language accurately."},
                    {"role": "user", "content": prompt}
    ]
    while True:
        try:
            message = call_chatgpt(conversations)
            json_text = parse_json(message)
            break
        except Exception as e:
            print(e)
            continue
        
    return json_text
        

       


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv', type=str)
    parser.add_argument('--outdir', type=str, default="output/coco2017")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    output_path = os.path.join(args.outdir, "chatgpt_lsg.json")
    samples = json.load(open(args.conv, "r"))
    
    print("starting conversation with model...")
    for sample in tqdm.tqdm(samples):
        lsg = parse_scene_graph(sample)
        sample["lsg"] = lsg
    
    
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=4)
