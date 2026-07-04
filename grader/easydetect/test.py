from pipeline.run_pipeline import *
import argparse
import tqdm
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--json", type=str)
parser.add_argument("--outfile", type=str)
args = parser.parse_args()

def parse_json(text):
    pattern = r"```json(.*)```"
    match = re.search(pattern, text, re.DOTALL)
    json_text = match.group(1) if match else text
    return json.loads(json_text)

pipeline = Pipeline()
samples = json.load(open(args.json, "r"))

for sample in tqdm.tqdm(samples):
    conversations = sample["conversations"]
    image_path = os.path.join("../../data/vg/", sample["url"].split("/")[-2], sample["url"].split("/")[-1])
    for conv in conversations:
        ret_list = []
        for sentence in conv["response"].split("."):
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue
            while True:
                try:
                    response, claim_list = pipeline.run(text=sentence, image_path=image_path, type="image-to-text")
                    print(response)
                    response = parse_json(response)
                    for ret, claim in zip(response, claim_list):
                        claim_content = claim.split(":")[-1]
                        claim_name = claim.split(":")[0]
                        assert claim_name in ret.keys()
                        claim_result = ret[claim_name]
                        reason = ret["reason"]
                        
                        ret_dict = {
                            "claim": claim_content,
                            "reason": reason,
                            "claim_result": claim_result,
                            "sentence": sentence
                        }
                        ret_list.append(ret_dict)
                except Exception as e:
                    print(e)
                    import traceback
                    traceback.print_exc()
                    continue
                
                break
            
        conv["easydetect_result"] = ret_list


with open(args.outfile, "w") as f:
    json.dump(samples, f, indent=4, ensure_ascii=False)
            
            

