import os
import json
import tqdm
import argparse
from PIL import Image
from openai import OpenAI
import time

from shr_utils import *

from utils.vg import format_case_vg
from utils.coco import format_case_coco

from dotenv import load_dotenv

load_dotenv(".env")

GPT_JUDGE_PROMPT = '''
Please help me judge if the comment of this image is hallucination or correct. 
I will give you a list of factual information about the image content. 

* **An image** (represented as a list of objects, their attributes, and relationships).
* **Bounding-box coordinates** for each object, given as `(x1, y1, x2, y2)` in normalized values between 0 and 1, corresponding to top-left and bottom-right corners.
* **Region descriptions** for each region, given as `(x1, y1, x2, y2): region description`, where `(x1, y1, x2, y2)` is the bounding box of the region in normalized values between 0 and 1, corresponding to top-left and bottom-right corners, and `region description` is the description of the region.

I will also give you a list of comments of the image for you to judge if it is hallucination based on the given factual information and image content. Please give a judgement one by one along with the reason.

Your output should be:
Judgement:
1. hallucination or correct or cannot judge: <reason>
2. ...


Factual Image Information:
{}

Here is the comment for you to judge (hallucination, correct, or cannot judge): 
{}
'''

def parse_args():
    parser = argparse.ArgumentParser(description="SHR Evaluation")
    # parser.add_argument("--api-key", type=str, required=True, help="key to the OPENAI API.")
    parser.add_argument("json_file", type=str, help="path to the json file, where model responses are stored.")
    parser.add_argument('--outdir', type=str, default=None, help='GPT-4 evaluation results to be saved')
    parser.add_argument('--api-url', type=str, default=None,
                       help='API endpoint URL (e.g., http://localhost:8003/v1). Overrides OPENAI_BASE_URL env var.')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API authorization key. Overrides OPENAI_API_KEY env var.')
    parser.add_argument('--gpt-model', type=str, default='gpt-4o', help='Model name to use for evaluation')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    
    # Setup OpenAI client - use args if provided, otherwise fall back to env vars
    api_key = args.api_key if args.api_key else os.getenv("OPENAI_API_KEY")
    base_url = args.api_url if args.api_url else os.getenv("OPENAI_BASE_URL")
    
    # Initialize OpenAI client
    # If base_url is provided, ensure it ends with /v1 (OpenAI client adds /chat/completions)
    if base_url:
        if base_url.endswith('/chat/completions'):
            base_url = base_url.replace('/chat/completions', '')
        if not base_url.endswith('/v1'):
            base_url = base_url.rstrip('/') + '/v1'
    
    # Initialize client - only pass parameters that are set
    # For local vLLM (base_url set but no api_key), use a dummy key (vLLM ignores it)
    client_kwargs = {}
    if base_url:
        client_kwargs['base_url'] = base_url
        # Local vLLM doesn't require API key, but OpenAI client needs one, so use dummy
        client_kwargs['api_key'] = api_key if api_key else "dummy-key"
    elif api_key:
        client_kwargs['api_key'] = api_key
    
    client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
    
    # json file to be evaluated
    records = json.load(open(args.json_file))
    
    judgement = {}
    run_all = ['run1']
    for run in run_all:
        judgement[run] = {}

    for i, record in enumerate(tqdm.tqdm(records)):
        image_info = format_case_vg(record, use_region=True)
        model_response = " ".join([conv["response"] for conv in record["conversations"]])
        model_cap_sep, is_repeated = get_model_cap(model_response)

        judge_prompt = GPT_JUDGE_PROMPT.format(image_info, model_cap_sep)
        
        for run in run_all:
            while True:
                try:
                    judge = client.chat.completions.create(
                        model=args.gpt_model,
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