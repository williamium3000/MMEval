import json
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("infile", type=str)

args = argparser.parse_args()

data = json.load(open(args.infile, "r"))

sample_count = 0
hallucinated_sample_count = 0

sentence_count = 0
hallucinated_sentence_count = 0

hallucination_count = 0

for sample in data[:50]:
    conversations = sample["conversations"]
    has_hallucination = False
    sample_count += 1
    for conv in conversations:
        for sentence in conv["respond_sentences"]:
            sentence_count += 1
            if len(sentence["hallucinations"]) > 0:
                has_hallucination = True
                
                hallucinated_sentence_count += 1
                hallucination_count += len(sentence["hallucinations"])
    if has_hallucination:
        hallucinated_sample_count += 1

print("sample hallucination rate: {:.2f}%".format(hallucinated_sample_count / sample_count * 100))
print("sentence hallucination rate: {:.2f}%".format(hallucinated_sentence_count / sentence_count * 100))
print("hallucination per sentence: {:.2f}".format(hallucination_count / hallucinated_sentence_count))