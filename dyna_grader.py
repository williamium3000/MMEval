from dyna.utils import call_chatgpt
from dyna.prompt import GRADER_PROMPT
import os
import argparse
import json
import tqdm
import re


def parse_json(text):
    pattern = r"```[(.*)]```"
    match = re.search(pattern, text, re.DOTALL)
    json_text = match.group(1) if match else text
    return json.loads(json_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv', type=str)
    parser.add_argument('--outdir', type=str, default="output/coco2017")
    args = parser.parse_args()

    samples = json.load(open(args.conv, "r"))
    for sample in tqdm.tqdm(samples):
        conversation = sample.pop("conversation")
        conversation_to_be_evaluated = []
        for turn in conversation:
            if turn["role"] == "evaluatee":
                conversation_to_be_evaluated.append(turn["content"])
        grading_results = []
        for c in conversation_to_be_evaluated:
            prompt = GRADER_PROMPT.format(sample, c)
            prompt = [
                            {"role": "system", "content": "You are a brilliant hallucination judger."},
                            {"role": "user", "content": prompt}
            ]
            while True:
                try:
                    result = call_chatgpt(prompt)
                    parsed = parse_json(result)
                    break
                except Exception as e:
                    print(e)
            grading_results.extend(parsed)
        sample["grading_results"] = grading_results
        sample["conversation"] = conversation
        
    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, "grading.json")
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=4)
