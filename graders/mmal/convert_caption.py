import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('json', type=str, default='output/dyna_bad_examples/coverage_certainty_with_answer_json_mode.json', help='response file containing images, questions, and model responses')
args = parser.parse_args()

filename = args.json.replace(".json", "_converted-mmal.json")

data = json.load(open(args.json))

for sample in data:
    captions = " ".join(sample['captions'])
    sample["conversations"][0]["gt"] = captions
    
json.dump(data, open(filename, "w"), indent=4)