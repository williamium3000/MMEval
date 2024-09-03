import json
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("infile", type=str)

args = argparser.parse_args()

data = json.load(open(args.infile, "r"))

sample_count = 0
hallucinated_sample_count = 0

claim_count = 0
hallucinated_claim_count = 0

for sample in data:
    conversations = sample["conversations"]
    has_hallucination = False
    sample_count += 1
    for conv in conversations:
        for hallucination in conv["hallucinations"]:
            claim_count += 1
            keys = list(hallucination.keys())
            assert "claim" in keys[0], hallucination
            if "non" in hallucination[keys[0]]:
                pass
            else:
                has_hallucination = True
                hallucinated_claim_count += 1
            
    if has_hallucination:
        hallucinated_sample_count += 1

print("sample hallucination rate: {:.2f}%".format(hallucinated_sample_count / sample_count * 100))
print("claim hallucination rate: {:.2f}%".format(hallucinated_claim_count / claim_count * 100))
