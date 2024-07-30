import argparse
import json
import random
import os
import pandas as pd

# convert output format to chair format
# add caption key and set as output
parser = argparse.ArgumentParser()
parser.add_argument("json", type=str)
args = parser.parse_args()

samples = json.load(open(args.json, 'r'))
for sample in samples:
    responses = [conv["response"] for conv in sample["conversations"]]
    sample["caption"] = " ".join(responses)

filename = os.path.basename(args.json)

json.dump(samples, open(os.path.join("eval/chair/output", filename), 'w'), indent=4)
