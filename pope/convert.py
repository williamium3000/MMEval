import argparse
import json
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument("json")
args = parser.parse_args()

all_data = [json.loads(q) for q in open(args.json, 'r')]

ret = []
for i in range(len(all_data)):
    sample = all_data[i]
    sample["question"] = sample["text"]
    sample["image"] = f'val2014/{sample["image"]}'
    ret.append(sample)

filename = args.json.split(".")[0] + "_converted.json"
json.dump(ret, open(filename, 'w'), indent=4)
