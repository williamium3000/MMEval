import os
import json
import tqdm
import argparse
from PIL import Image
import openai
import time

from shr_utils import *


from dotenv import load_dotenv

load_dotenv(".env")

GPT_JUDGE_PROMPT = '''
Please help me judge if the comment of this image is hallucination or correct. 
I will give you a list of region description of a image. The format is [x1, y1, x2, y2]: region description, where [x1, y1, x2, y2] is the bounding box of the region. This is the ground truth information of the image. Besides, I give you some factual information about the content of the image (which is 100% accurate). Your judgement should base on this information. However, this information only descibe the objects in the region of image, so it cannot descibe the subjective part of the image, e.g., atmosphere, style, emotion. In that case, you can return "Cannot judge".
Also, I will give you a list of comments of the image for you to judge if it is hallucination. Please give a judgement one by one along with the reason.

Your output should be:
Judgement:
1. hallucination or correct or cannot judge: <reason>
2. ...

Here are the region descriptions of the image:
{}

Factual Information:
{}

Here is the comment for you to judge (hallucination, correct, or cannot judge): 
{}
'''

def parse_args():
    parser = argparse.ArgumentParser(description="SHR Evaluation")
    # parser.add_argument("--api-key", type=str, required=True, help="key to the OPENAI API.")
    parser.add_argument("json_file", type=str, help="path to the json file, where model responses are stored.")
    parser.add_argument('--outdir', type=str, default=None, help='GPT-4 evaluation results to be saved')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    
    # setup openai
    # setup_openai(args.api_key)
    
    # json file to be evaluated
    records = json.load(open(args.json_file))
    
    judgement = {}
    run_all = ['run1']
    for run in run_all:
        judgement[run] = {}

    for i, record in enumerate(tqdm.tqdm(records)):

        description = ""
        instances = record["instances"]
        for ins in instances:
            description += f"{ins['bbox']}: {ins['category']}\n"
        
        model_response = " ".join([conv["response"] for conv in record["conversations"]])
        model_cap_sep, is_repeated = get_model_cap(model_response)

        judge_prompt = GPT_JUDGE_PROMPT.format(description, "\n".join(record["captions"]), model_cap_sep)
        
        for run in run_all:
            while True:
                try:
                    judge = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful, impartial and objective judge that can accurately evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question."},
                            {"role": "user", "content": judge_prompt}
                        ],
                        temperature=0.0,
                    ).choices[0].message.content
                except Exception as e:
                    print(e)
                    print('retrying...')
                    time.sleep(10)
                    continue
                if "judgement" in judge.lower():
                    break
            # post-process
            final_judge = post_process_no_revise(judge, model_response)
            judgement[run][i] = {
                "raw_judgement": judge,
                "mode_response": model_response,
                "judgement": final_judge,
            }
        
    whole_sample_cnt = len(records)
   
    os.makedirs(args.outdir, exist_ok=True)
    localtime = time.asctime( time.localtime(time.time()) ).replace(' ', '_')
    # save metrics
    metrics = {}
    for run in run_all:
        metrics[run] = {}
        get_metric(judgement[run], metrics[run])
    
    print(metrics)
    # halucination ratio
    metrics["mean_hal_ratio"] = round(
        sum(metrics[run]["hal_sents_ratio"] for run in run_all)/len(run_all), 3
    )
    # dump judgement file
    with open(os.path.join(args.outdir, 'judgement.json'), "w") as f:
        json.dump(judgement, f)
    # dump metric file
    with open(os.path.join(args.outdir, 'metrics.json'), "w") as f:
        json.dump(metrics, f)