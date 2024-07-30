import os
import sng_parser
import argparse
from pprint import pprint
import nltk
from nltk.corpus import wordnet
import tqdm
import json

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def is_concrete(word):
    synsets = wordnet.synsets(word, pos='n')
    if not synsets:
        return False
    # A simple heuristic: check if the first sense of the word is concrete based on WordNet
    allowed = ["noun.animal", "noun.artifact", "noun.body", "noun.food", 'noun.object', "noun.person", 'noun.plant', '']
    return any(str(synset.lexname()) in allowed for synset in synsets)



filter_attribute = ['a', 'the', 'an', '', 'this', 'other', "these", 'exact']
filter_instance = ['image', '', 'this', 'they', 'it', 'atmosphere', 'scene', 'part', 'side', 'number', 'presence', 'space']
fliter_relation = ['including']

def reformat_graph(graph):
    
    entities = graph['entities']
    relations = graph['relations']

    instances = []
    attributes = []
    for e in entities:
        entity_name = e['head'].lower()
        if entity_name in filter_instance or not is_concrete(entity_name):
            continue
        instances.append(entity_name)
        
        for x in e['modifiers']:
            attribute = x['span'].lower()
            if attribute not in filter_attribute:
                attributes.append((attribute, entity_name))

    relations = [
        [
            entities[rel['subject']]['head'].lower(),
            rel['relation'].lower(),
            entities[rel['object']]['head'].lower()
        ]
        for rel in relations
    ]
    filtered_relations = []
    for a, rel, b in relations:
        if a not in instances:
            continue
        if rel in fliter_relation:
            continue
        filtered_relations.append((a, rel, b))
        
    return list(set(instances)), list(set(attributes)), list(set(filtered_relations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv', type=str)
    parser.add_argument('--outdir', type=str)
    
    args = parser.parse_args()
    data = json.load(open(args.conv, "r"))
    for sample in tqdm.tqdm(data):
        conversation = sample["conversation"]
        conversation_to_be_evaluated = []
        for turn in conversation:
            if turn["role"] == "evaluatee":
                conversation_to_be_evaluated.append(turn["content"])
        conversation = " ".join(conversation_to_be_evaluated)
    
        sg = sng_parser.parse(conversation)
        instances, attributes, relations = reformat_graph(sg)
        sample["lsg"] = {
            "instances": instances,
            "attributes": attributes,
            "relations": relations
        }
    
    output_path = os.path.join(args.outdir, "parser_lsg.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


  