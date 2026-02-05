from nltk.stem import *
from nltk.corpus import wordnet as wn
import json
import argparse
import os
import json
import numpy as np
import nltk
import re

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import tqdm

ps = PorterStemmer()
def sentence2words(sentence):
    phrases = ["cell phone"]
    filtered_non_phy_words = ["side", "city", "image", "addition"]
    # 1. Lowercasing
    sentence = sentence.lower()

    # 2. Replacing MWEs with underscores before tokenization
    for phrase in phrases:
        sentence = sentence.replace(phrase, phrase.replace(" ", "_"))

    # 3. Removing Punctuation
    sentence = re.sub(r'[^\w\s]', '', sentence)

    # 4. Tokenization
    tokens = word_tokenize(sentence)

    # 5. POS Tagging to filter only Nouns (NN, NNS, NNP, NNPS)
    pos_tags = nltk.pos_tag(tokens)
    nouns = [word for word, pos in pos_tags if pos.startswith('NN') or '_' in word]  # Keep MWEs regardless of POS
    nouns = [word for word in nouns if word not in filtered_non_phy_words]
    # 6. Removing Stop Words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in nouns if word not in stop_words]
    
    # 7. Stemming (excluding MWEs)
    stemmed_tokens = [ps.stem(word) if '_' not in word else word for word in filtered_tokens]

    # 8. Reverting underscores to spaces for readability
    final_tokens = [word.replace('_', ' ') for word in stemmed_tokens]
    
    return final_tokens


def match_vg_list(object_dict, words):
    match_result = []
    for vg_object in object_dict.keys():
        for word in words:
            if vg_object == word:
                match_result.append(vg_object)
    match_result = list(set([(object_dict[match][0], object_dict[match][1]) for match in match_result]))
    return match_result

def compute_chair(caps, total_synsets):

    '''
    Given ground truth objects and generated captions, determine which sentences have hallucinated words.
    '''

    num_caps = 0.
    num_hallucinated_caps = 0.
    hallucinated_word_count = 0.
    vg_word_count = 0.
    
    output = {
        'sentences': [], "vg_words": [], 
        "vg_hallucinated_words": [],
        "object_in_gts": [],
        "gt_objects": [],
        "coverage": []} 

    for i, cap_eval in enumerate(tqdm.tqdm(caps)):

        cap = cap_eval['caption']
        imid = cap_eval['image_id']

        gt_synsets = []
        # objects
        for obj in cap_eval['objects']:
            gt_synsets.extend(obj['synsets'])
        # attributes
        for attr in cap_eval['attributes']:
            gt_synsets.extend(attr['synsets'])
        # relationships
        for rel in cap_eval['relationships']:
            gt_synsets.extend(rel["subject"]['synsets'])
            gt_synsets.extend(rel["object"]['synsets'])
        for rcap in cap_eval['regions']:
            rcap = rcap['phrase']
            
            raw_words_recap = sentence2words(rcap)
            output_objects_recap = match_vg_list(total_synsets, raw_words_recap)
            synset_recap = [_[1] for _ in output_objects_recap]
            gt_synsets.extend(synset_recap)
            
        gt_synsets = list(set(gt_synsets))
        
        raw_words = sentence2words(cap)
        
        output_objects = match_vg_list(total_synsets, raw_words)
        
        
        cap_dict = {'image_id': cap_eval['image_id'], 
                    'caption': cap,
                    'vg_hallucinated_words': [],
                    'vg_gt_words': gt_synsets,
                    'vg_generated_words': [_[0] for _ in output_objects],
                    "words": raw_words
                    }
        # print(cap_eval)
        cap_dict['metrics'] = {'CHAIRs': 0,
                                'CHAIRi': 0}

        #count hallucinated words
        vg_word_count += len(raw_words)
        hallucinated = False
        vg_words_i = []
        vg_hallucinated_words_i = []
        object_in_gts = []
        
        for word, syn_class in output_objects:
            
            vg_words_i.append((word, syn_class))
            if is_hallucinate(syn_class, gt_synsets):
                hallucinated_word_count += 1 
                cap_dict['vg_hallucinated_words'].append((word, syn_class))
                
                vg_hallucinated_words_i.append((word, syn_class))
                hallucinated = True    
            else:
                object_in_gts.append(syn_class)

        vg_words_i = list(set(vg_words_i))
        vg_hallucinated_words_i = list(set(vg_hallucinated_words_i))
        
        output['vg_words'].extend(vg_words_i)
        output['vg_hallucinated_words'].extend(vg_hallucinated_words_i)
        output['object_in_gts'].extend(list(set(object_in_gts)))
        output['gt_objects'].extend(gt_synsets)
        
        if len(gt_synsets) > 0:
            output["coverage"].append(len(list(set(object_in_gts))) / len(gt_synsets))
        
        #count hallucinated caps
        num_caps += 1
        if hallucinated:
            num_hallucinated_caps += 1

        cap_dict['metrics']['CHAIRs'] = int(hallucinated)
        cap_dict['metrics']['CHAIRi'] = 0.
        if len(raw_words) > 0:
            cap_dict['metrics']['CHAIRi'] = len(cap_dict['vg_hallucinated_words'])/float(len(raw_words))

        output['sentences'].append(cap_dict)

    chair_s = (num_hallucinated_caps/num_caps)
    chair_i = (hallucinated_word_count/vg_word_count)
    chair_i_v2 = len(output['vg_hallucinated_words'])/len(output['vg_words'])
    coverage_avg = np.mean(output["coverage"])
    coverage_all = len(output['object_in_gts']) / len(output['gt_objects'])
    output['overall_metrics'] = {
                                    'CHAIRs': chair_s,
                                    'CHAIRi': chair_i,
                                    "CHAIRi_v2": chair_i_v2,
                                    "Coverage_avg": coverage_avg,
                                    "Coverage_all": coverage_all
                                    }

    return output 


def save_hallucinated_words(cap_file, cap_dict): 
    tag = cap_file.split('/')[-1] 
    with open(os.path.join(os.path.dirname(cap_file), f'hallucinated_words_{tag}'), 'w') as f:
        json.dump(cap_dict, f, indent=4)

def print_metrics(hallucination_cap_dict, quiet=False):
    sentence_metrics = hallucination_cap_dict['overall_metrics']
    metric_string = "%0.01f\t%0.01f\t%0.01f\t%0.01f\t%0.01f" %(
                                                    sentence_metrics['CHAIRs']*100,
                                                    sentence_metrics['CHAIRi']*100,
                                                    sentence_metrics['CHAIRi_v2']*100,
                                                    sentence_metrics['Coverage_avg']*100,
                                                    sentence_metrics['Coverage_all']*100
                                                  )

    if not quiet:
        print("CHAIRs\tCHAIRi\tCHAIRi_v2\tCoverage-avg\tCoverage-all")
        print(metric_string)

    else:
        return metric_string






def is_father_child_relationship(tree, father_name, child_name):
    if not tree:
        return False

    if tree["name"] == father_name:
        for child in tree["children"]:
            if child["name"] == child_name:
                return True

    for child in tree["children"]:
        if is_father_child_relationship(child, father_name, child_name):
            return True

    return False


def is_hallucinate(response_word, gt_words):
    
    response_synset = wn.synset(response_word)
    response_hyponyms = get_hyponyms_tree(response_synset)
    for gt_word in gt_words:
        gt_synset = wn.synset(gt_word)
        gt_hypernyms = get_hypernym_tree(gt_synset)

        if (response_word == gt_word) or (response_synset in gt_hypernyms) or (gt_synset in response_hyponyms):
            return False
    
    return True


def get_hypernym_tree(synset):
    hypernyms = set()

    def traverse(syn):
        if syn not in hypernyms:
            hypernyms.add(syn)
            for hypernym in syn.hypernyms():
                traverse(hypernym)

    traverse(synset)
    return hypernyms

def get_hyponyms_tree(synset):
    hyponyms = set()

    def traverse(syn):
        if syn not in hyponyms:
            hyponyms.add(syn)
            for hypernym in syn.hyponyms():
                traverse(hypernym)

    traverse(synset)
    return hyponyms

def is_physical_object(syn):
    syn = wn.synset(syn)
    hypernym_tree = get_hypernym_tree(syn)
    if any(hypernym.name().startswith(("physical_entity.n.", "people.n.", "vegetation.n")) for hypernym in hypernym_tree):
        return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cap_file", type=str)
    parser.add_argument("object_synsets", type=str)
    args = parser.parse_args()
    
    with open(args.object_synsets) as object_dict_file:
        object_dict = json.load(object_dict_file)
        object_dict = {key.split('.')[0]: value for key, value in object_dict.items()}

    object_dict = {key: value for key, value in object_dict.items() if is_physical_object(value)}
    
    stemmed_object_dict = {}
    
    for word, synset in object_dict.items():
        stemmed_object_dict[ps.stem(word)] = (word, synset)
    
    data = json.load(open(args.cap_file, 'r'))
    for sample in data:
        # Create caption from conversations
        if "conversations" in sample:
            responses = [conv["response"] for conv in sample["conversations"]]
            sample["caption"] = " ".join(responses)
        elif "caption" not in sample:
            raise ValueError(f"No caption found for sample {sample}")
        
        # Handle metadata: if metadata exists, copy fields to top level
        # Otherwise, assume fields are already at top level (like in caption files)
        if "metadata" in sample:
            for k, v in sample["metadata"].items():
                sample[k] = v
        
        # Ensure required fields exist (with defaults if missing)
        if "objects" not in sample:
            sample["objects"] = []
        if "attributes" not in sample:
            sample["attributes"] = []
        if "relationships" not in sample:
            sample["relationships"] = []
        if "regions" not in sample:
            sample["regions"] = []
        
    chair_result = compute_chair(data, stemmed_object_dict)
    print_metrics(chair_result)
    save_hallucinated_words(args.cap_file, chair_result)