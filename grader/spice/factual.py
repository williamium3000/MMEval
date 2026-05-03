import argparse
import json
import os
import tqdm
import torch
from nltk.wsd import lesk
#from factual_scene_graph.evaluation.evaluator import Evaluator
#from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
import re
from nltk import word_tokenize, pos_tag, RegexpParser
from nltk.corpus import wordnet as wn

# import nltk
# nltk.download('wordnet')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')

from grader.claim_level.claim_extraction import CLAIM_PROMPT
from utils.construct_tree import get_hypernym_tree, get_object_map
from utils.llm import LLMChat

filtered_non_phy_words = ["city", "image", "city life", "landscape", "light", "time", "diversity", 'traffic rule', 'cityscape','activity','scene','environment', 'workspace', 'area']
filtered_attribute = ["visible", 'well-organized']


def is_physical_word(subject_word, sg_tuple, object_dict):
    if subject_word.lower() in filtered_non_phy_words:
        return False
    elif subject_word.lower() in object_dict.keys() or subject_word in object_dict.values():
        return True
    else:
        phrase = sg_tuple.replace(',', '').lower()
        best_synset = lesk(phrase, subject_word, pos=wn.NOUN)
        if best_synset:
            syn = wn.synset(best_synset.name())
            hypernym_tree = get_hypernym_tree(syn)
            if any(hypernym.name().startswith(
                    ("physical_entity.n.", "people.n.", "person.n.", "vegetation.n", "building.n", "tree.n")) for
                   hypernym in hypernym_tree):
                return True

        for word in subject_word.split(' '):
            best_synset = lesk(phrase, word, pos=wn.NOUN)

            if best_synset:
                syn = wn.synset(best_synset.name())
                hypernym_tree = get_hypernym_tree(syn)
                if any(hypernym.name().startswith(("physical_entity.n.", "people.n.", "person.n.", "vegetation.n", "building.n", "tree.n")) for hypernym in hypernym_tree):
                    # print(True, syn)
                    return True
        # print(False, syn)
        return False


if __name__ == '__main__':
    # sentence_list = ['There is no car on the street.']
    # sg_model = 'lizhuang144/flan-t5-small-VG-factual-sg'
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    #
    # parser = SceneGraphParser(sg_model, device=device)
    # text_graph = parser.parse(sentence_list, beam_size=1, return_text=True, max_input_len=1000,
    #                           batch_size=64)
    #
    # print(text_graph)


    parser = argparse.ArgumentParser()
    parser.add_argument('--conv', type=str, default="output/vg/caption.json")
    parser.add_argument('--sg_model', type=str, default='lizhuang144/flan-t5-small-VG-factual-sg')
    parser.add_argument('--text_encoder', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--metric', type=str, default='all', choices=['all', 'set_match', 'spice', 'soft_spice'])
    parser.add_argument('--outdir', type=str, default='output/vg/sg')

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    output_path = os.path.join(args.outdir, "icl_caption_lsg.json")

    object_dict = get_object_map(object_file_path='graders/chair/data/filtered_object_synsets_final.json')

    # Load parser and evaluator
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # parser = SceneGraphParser(args.sg_model, device=device)
    # evaluator = Evaluator(parser=parser, text_encoder_checkpoint=args.text_encoder, device=device, lemmatize=False)
    # agent = LLMChat("gpt-4o")
    #
    # # Load conversation
    # conversation_list = []
    # ref_sg_list = []
    # samples = json.load(open(args.conv, "r"))
    # for sample in tqdm.tqdm(samples):
    #     conversation = sample["conversations"]
    #     sentence_list = []
    #     for turn in conversation:
    #         sentence_list += [sent for sent in turn["response"].strip().split('.') if len(sent)>0]
    #
    #     # Extract claim
    #     sentence_list = []
    #     response_list = [conversation["response"].strip() for conversation in sample["conversations"]]
    #     responses = " ".join(response_list)
    #
    #     # for responses in response_list:
    #     #     input_to_gpt = [
    #     #         {"role": "system", "content": f"You are a brilliant claim generator. {CLAIM_PROMPT}"},
    #     #         {"role": "user", "content": responses}
    #     #     ]
    #     #     claim_list = agent.chat(input_to_gpt, None, temperature=0)
    #     #     claim_json = json.loads(claim_list)
    #     #
    #     #     for claim_dict in claim_json:
    #     #         sentence_list += [claim["claim"] for claim in claim_dict["claims"]]
    #
    #     input_to_gpt = [
    #             {"role": "system", "content": f"You are a brilliant claim generator. {CLAIM_PROMPT}"},
    #             {"role": "user", "content": responses}
    #         ]
    #     claim_list = agent.chat(input_to_gpt, None, temperature=0)
    #     claim_json = json.loads(claim_list)
    #
    #     for claim_dict in claim_json:
    #         sentence_list += [claim["claim"] for claim in claim_dict["claims"]]
    #
    #     sent_num = len(sentence_list)
    #
    #     # Add regional info
    #     # sentence_list += [des["phrase"] for des in sample['regions']]
    #
    #     text_graph = parser.parse(sentence_list, beam_size=args.beam_size, return_text=True, max_input_len=1000, batch_size=args.batch_size)
    #
    #     # Check results
    #     # for sentence, graph in zip(sentence_list, text_graph):
    #     #     print(sentence, graph)
    #
    #     # Add regional info
    #     # ref_sg = text_graph[sent_num:]
    #     # reformatted_sg = ' , '.join(ref_sg)
    #
    #     ref_sg = []
    #
    #     # Add relationship info
    #     for rel in sample["relationships"]:
    #         rel_info = f"( {rel['subject']['names'][0].lower()} , {rel['predicate'].lower()} , {rel['object']['names'][0].lower()} )"
    #         ref_sg += [rel_info]
    #
    #     # Add attribute info
    #     for attr in sample["attributes"]:
    #         if attr["attributes"] is None or len(attr["attributes"]) == 0:
    #             # sentence_list += [f"( {attr['names'][0]} )"]
    #             continue
    #         else:
    #             for attribute in attr['attributes']:
    #                 att_info = f"( {attr['names'][0]}, is , {attribute} )"
    #                 ref_sg += [att_info]
    #
    #     # Add object num
    #     object_num_dict = {}
    #     for object_name in sample["objects"]:
    #         # if object_name["names"][0] not in object_num_dict.keys():
    #         #     object_num_dict[object_name["names"][0]] = 1
    #         # else:
    #         #     object_num_dict[object_name["names"][0]] = +1
    #
    #         ref_sg += [f'( {object_name["names"][0]} )']
    #
    #     # for key, value in object_num_dict.items():
    #     #     if value > 1:
    #     #         ref_sg += [f'( {key} , is , {value} )']
    #     ref_sg_list.append([' , '.join(set(ref_sg))])
    #
    #     conv_text_graph = ' , '.join(text_graph[:sent_num])
    #
    #     # Filter out non-physical noun words
    #     filtered_conv = []
    #     for sg_tuple in re.findall(r'\( (.*?) \)', conv_text_graph):
    #         word_list = sg_tuple.split(' , ')
    #         if len(word_list) == 1:
    #             subject = word_list[0]
    #             if is_physical_word(subject, sg_tuple, object_dict):
    #                 filtered_conv.append(f'( {sg_tuple} )')
    #             else:
    #                 print('filtered:', sg_tuple)
    #         elif 'is' in word_list:
    #             subject = word_list[0]
    #             attribute = word_list[2]
    #             is_physical = is_physical_word(subject, sg_tuple, object_dict)
    #             if is_physical and not re.search(r'\d', attribute) and attribute not in filtered_attribute:
    #                 filtered_conv.append(f'( {sg_tuple} )')
    #             elif is_physical:
    #                 filtered_conv.append(f'( {subject} )')
    #             else:
    #                 print('filtered:', sg_tuple)
    #         else:
    #             subject = word_list[0]
    #             object_word = word_list[2]
    #             subject_status = is_physical_word(subject, sg_tuple, object_dict)
    #             object_status = is_physical_word(object_word, sg_tuple, object_dict)
    #             if subject_status and object_status:
    #                 filtered_conv.append(f'( {sg_tuple} )')
    #             elif object_status:
    #                 filtered_conv.append(f'( {object_word} )')
    #             elif subject_status:
    #                 filtered_conv.append(f'( {subject} )')
    #             else:
    #                 print('filtered:', sg_tuple)
    #
    #     conversation_list.append(' , '.join(set(filtered_conv)))
    #
    #     # print(conversation_list[0])
    #     # print(ref_sg_list[0])
    #
    #     sample["lsg"] = ' , '.join(set(filtered_conv))
    #     with open(output_path, "w") as f:
    #         json.dump(samples, f, indent=4)

    # Evaluate
    # spice_scores, cand_graphs, ref_graphs = evaluator.evaluate(
    #     conversation_list,
    #     ref_sg_list,
    #     method='spice',
    #     beam_size=args.beam_size,
    #     batch_size=128,
    #     max_input_len=10000,
    #     max_output_len=256,
    #     return_graphs=True
    # )
    # print('SPICE scores:', sum(spice_scores) / len(spice_scores))
    #
    # if args.metric == 'all':
    #     # set_match_scores = evaluator.evaluate(cand_graphs, ref_graphs, method='set_match', beam_size=1)
    #     # print('Set Match scores:', sum(set_match_scores) / len(set_match_scores))
    #
    #     soft_spice_scores = evaluator.evaluate(cand_graphs, ref_graphs, method='soft_spice', beam_size=1)
    #     print('Soft-SPICE scores:', sum(soft_spice_scores) / len(soft_spice_scores))
    # else:
    #     soft_spice_scores = evaluator.evaluate(cand_graphs, ref_graphs, method=args.metric, beam_size=1)
    #     print(f'{args.metric} scores:', sum(soft_spice_scores) / len(soft_spice_scores))
    #
    # for i in range(len(samples)):
    #     text_sg = cand_graphs[i]
    #     samples[i]["lsg"] = text_sg
    #     samples[i]["spice"] = spice_scores[i]
    #     samples[i]["soft_spice"] = soft_spice_scores[i]
    #
    # with open(output_path, "w") as f:
    #     json.dump(samples, f, indent=4)
