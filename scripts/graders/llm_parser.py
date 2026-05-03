import os
import argparse
import json
import glob
from collections import defaultdict
import numpy as np
import tqdm
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    torch = None
    device = "cpu"
# Lazy imports — these are only needed for the single-file LLM parsing path,
# not for the --sg_dir SoftSPICE scoring path used by run_all_metrics.sh.
try:
    from utils.construct_tree import get_object_map
    from graders.spice.factual import is_physical_word
    from utils.llm import LLMChat
    object_dict = get_object_map(object_file_path='data/filtered_object_synsets_final.json')
except Exception as _e:
    get_object_map = None
    is_physical_word = None
    LLMChat = None
    object_dict = {}

filtered_non_phy_words = ["city", "image", "city life", "landscape", "light", "time", "diversity", 'traffic rule', 'cityscape','activity','scene','environment', 'workspace', 'area']
filtered_attribute = ["visible", 'well-organized']


def get_hypernym_tree(synset):
    hypernyms = set()

    def traverse(syn):
        if syn not in hypernyms:
            hypernyms.add(syn)
            for hypernym in syn.hypernyms():
                traverse(hypernym)

    traverse(synset)
    return hypernyms


def refine_output(response):
    before_match_parsing = []
    if 'none' in response.lower():
        return []

    response = response.replace('>,', '>')
    for a in response.split("<")[1:]:
        a = a.split(",")
        # if len(a) < 3:
        #     continue
        if len(a) == 3 and a[1].strip() not in ['is', 'was', 'are', 'were']:
            sub, pred, obj = a[0].strip(), a[1].strip(), a[2].split(">")[0].strip()
            sub_status = is_physical_word(sub, ' '.join(a), object_dict)
            obj_status = is_physical_word(obj, ' '.join(a), object_dict)
            if sub_status and obj_status:
                before_match_parsing.append([sub, pred, obj])
            elif obj_status:
                before_match_parsing.append([obj])
            elif sub_status:
                before_match_parsing.append([sub])

        elif len(a) == 1:
            sub = a[0].split(">")[0].strip()
            if is_physical_word(sub, ' '.join(a), object_dict):
                before_match_parsing.append([sub])
        elif len(a) == 3 and a[1].strip() in ['is', 'was', 'are', 'were']:
            sub, pred, adj = a[0].strip(), a[1].strip(), a[2].split(">")[0].strip()
            if is_physical_word(sub, ' '.join(a), object_dict):
                before_match_parsing.append([sub, pred, adj])
        else:
            print('Error format', a)

    return before_match_parsing


def parse_scene_graph(sample, agent):
    conversation = sample["conversations"]

    output_dict = {}
    unique_sg = []
    for index, turn in enumerate(conversation):
        vlm_response = turn["response"]

        response_list = [question for question in vlm_response.strip('\n').split('\n') if len(question)]

        lsg_list = []
        for question in response_list:
            LSG_PROMPT = f"""From the given sentence, the task is to extract scene graphs formed as <subject, predicate, object>, <object, is, attribute> or <object>. Note that the subject is the physical entity or noun that performs the action or is being described, and the object is the physical entity or noun that is affected by the action or is receiving the action. The predicate is a verb or adjective without auxiliary verb, and is represented without the tense (e.g., are, being). The attribute is a physical quality or characteristic (typically an adjective) directly modifying an object or entity (e.g., <jacket, is, red>, <wall, is, wooden>).
Instructions:
- If an object has no attributes or relations, output it directly in the form <object>.
- Do **not** extract scene graphs involving:
  - Objects, subjects or relations that are negated (e.g., "There is no man...")
  - Non-physical entities (e.g., "atmosphere", "conversation") in subject or object
  - Entities or relations that are **speculative or inferred** from other clues rather than explicitly described as visible (e.g., "could indicate", "might suggest", "possibly", "likely")
  - Abstract scene descriptions that cannot be directly grounded in physical objects or traits (e.g., "scene is urban", "shirt adds pop of color", "building contributes to atmosphere")
  - Attributes that are subjective or stylistic rather than physical (e.g., "beautiful", "cozy", "futuristic" when not tied to tangible features)
  - Statements about effects, purposes, or benefits rather than direct physical description (e.g., "contributes to convenience", "supports community well-being")

### Examples
Sentence: "A slice of bread is covered with a sour cream and guacamole."
Triplets: <bread, covered with, sour cream>, <bread, covered with, guacamole>

Sentence: "A beautiful woman walking a dog on top of a beach."
Triplets: <woman, walking with, dog>, <woman, on, beach>, <dog, on, beach>

Sentence: "Four clocks sitting on a floor next to a woman's feet."
Triplets: <clock, sitting on, floor>, <clock, next to, feet>

Sentence: "One person sits in a chair looking at her phone while another rests on the couch."
Triplets: <person, sits in, chair>, <person, looking at, phone>, <person, rests on, couch>

Sentence: "A lady and a child near a park bench with kites and ducks flying in the sky and on the ground."
Triplets: <lady, near, park bench>, <child, near, park bench>, <kites, flying in, sky>, <ducks, on, ground>

Sentence: "Two men sit on a bench near the sidewalk and one of them talks on a cell phone."
Triplets: <men, sit on, bench>, <bench, near, sidewalk>, <man, talks on, phone>

Sentence: "There is no man wearing a red jacket in the image."  
Triplets: (none)

Sentence: "A man wearing a red jacket is in the image."  
Triplets: <man, wearing, jacket>, <jacket, is, red>

Sentence: "The carpet on the wooden floor is blue."  
Triplets: <carpet, on, floor>, <floor, is, wooden>, <carpet, is, blue>

Sentence: "There are several cars parked along the street, and a bicycle is also visible."  
Triplets: <cars, parked on, street>, <bicycle>

Sentence: "People on the street are engaged in various activities and interactions."  
Triplets: <people, on, street>

Sentence: "The traffic light indicates that the street is regulated for vehicle and pedestrian safety."  
Triplets: <traffic light>, <street>

### Now extract triplets from the following sentence:
Sentence: \"{question}\"\nTriplets:
            """

            messages = [
                        {"role": "system", "content": "From the given sentence, your task is to extract meaningful triplets formed as <subject, predicate, object>."},
                        {"role": "user", "content": LSG_PROMPT.strip()}
            ]

            parsing_result = agent.chat(messages, None)
            # print('SG', parsing_result)
            refined_sg = refine_output(parsing_result)

            lsg_list += refined_sg
            unique_sg += [f"( {' , '.join(sg)} )" for sg in refined_sg]
            # print('Question:', question)
            # print('Unique SG', [f"( {' , '.join(sg)} )" for sg in refined_sg])
        output_dict[index] = lsg_list
        
    return output_dict, set(unique_sg)


def get_anno_sg(sample):
    ref_sg = []
    # Add relationship info
    if "sg" in sample:
        for rel in sample["sg"]["relationships"]:
            rel_info = f"( {rel['subject']['names'][0].lower()} , {rel['predicate'].lower()} , {rel['object']['names'][0].lower()} )"
            ref_sg += [rel_info]

        # Add attribute info
        for attr in sample["sg"]["objects"].values():
            if attr["attributes"] is None or len(attr["attributes"]) == 0:
                continue
            else:
                for attribute in attr['attributes']:
                    att_info = f"( {attr['names'][0]}, is , {attribute} )"
                    ref_sg += [att_info]

        # Add object num
        object_num_dict = {}
        for object_name in sample["sg"]["objects"].values():
            # if object_name["names"][0] not in object_num_dict.keys():
            #     object_num_dict[object_name["names"][0]] = 1
            # else:
            #     object_num_dict[object_name["names"][0]] = +1

            ref_sg += [f'( {object_name["names"][0]} )']
    else:
        try:
            for rel in sample["relationships"]:
                rel_info = f"( {rel['subject']['names'][0].lower()} , {rel['predicate'].lower()} , {rel['object']['names'][0].lower()} )"
                ref_sg += [rel_info]

            # Add attribute info
            for attr in sample["attributes"]:
                if attr["attributes"] is None or len(attr["attributes"]) == 0:
                    continue
                else:
                    for attribute in attr['attributes']:
                        att_info = f"( {attr['names'][0]}, is , {attribute} )"
                        ref_sg += [att_info]

            # Add object num
            object_num_dict = {}
            for object_name in sample["objects"]:
                # if object_name["names"][0] not in object_num_dict.keys():
                #     object_num_dict[object_name["names"][0]] = 1
                # else:
                #     object_num_dict[object_name["names"][0]] = +1

                ref_sg += [f'( {object_name["names"][0]} )']
        except:
            ref_sg =[]

    return ref_sg


def _norm_sample_for_get_anno_sg(sample):
    """部分数据没有顶层 attributes，用 objects 顶替，便于 get_anno_sg 使用。"""
    if "attributes" not in sample and "objects" in sample:
        return {**sample, "attributes": sample["objects"]}
    return sample


def _collect_sg_inputs(samples, agent=None, sample_num=None, do_parse=False):
    """从 samples 得到 anno_conv_list 和 ref_sg_list。若 do_parse 且 agent 给定则调用 parse；否则要求样本已有 parsed_sg/unique_sg。
    Evaluator 要求：candidates 为 list of strings（每条样本一个字符串）；references 为 list of lists of strings。"""
    if sample_num is not None:
        samples = samples[:sample_num]
    ref_sg_list = []
    anno_conv_list = []
    for sample in samples:
        if do_parse and agent is not None and ("parsed_sg" not in sample or "unique_sg" not in sample):
            sg_dict, unique_sg = parse_scene_graph(sample, agent)
            sample["parsed_sg"] = sg_dict
            sample["unique_sg"] = list(unique_sg)
        else:
            unique_sg = sample.get("unique_sg") or []
            if isinstance(unique_sg, set):
                unique_sg = list(unique_sg)
        # candidates: list of strings（每条样本一个字符串）
        anno_conv_list.append(' , '.join(str(s) for s in unique_sg))
        ref_sg = get_anno_sg(_norm_sample_for_get_anno_sg(sample))
        # references: list of lists of strings
        ref_sg_list.append([str(s) for s in set(ref_sg)])
    return anno_conv_list, ref_sg_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv_script', type=str, default="output/five_context/InternVL3-8B-Instruct_merged_internal.json", help='单个输入 json（与 --sg_dir 二选一）')
    parser.add_argument('--sg_dir', type=str, default=None, help='输入目录：对该目录下所有 .json 批量算 SPICE 并输出总结（不强制文件名尾缀）')
    parser.add_argument('--sg_model', type=str, default='lizhuang144/flan-t5-small-VG-factual-sg', help='Evaluator 用到的场景图解析模型')
    parser.add_argument('--metric', type=str, default='all', choices=['all', 'set_match', 'spice', 'soft_spice'])
    parser.add_argument('--text_encoder', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sample_num', type=int, default=100, help='单文件模式下最多用多少条样本')
    parser.add_argument('--outdir', type=str, default="output/five_context/")
    args = parser.parse_args()

    if args.sg_dir:
        # Self-contained scorer: SPICE = exact-match set F1 over normalized
        # triple strings; Soft-SPICE = mean-of-max cosine F1 via sentence-
        # transformers. Reuses the SG grader's LLM-parsed `unique_sg` directly,
        # so factual_scene_graph's SceneGraphParser is NOT invoked.
        from sentence_transformers import SentenceTransformer

        def _norm_triple(s):
            if not isinstance(s, str):
                return None
            t = s.strip()
            if t.startswith('('): t = t[1:]
            if t.endswith(')'):   t = t[:-1]
            parts = [p.strip().lower() for p in t.split(',') if p.strip()]
            return ', '.join(parts) if parts else None

        def _hard_f1(c, r):
            if not c and not r: return 1.0
            if not c or not r:  return 0.0
            inter = len(c & r)
            if inter == 0: return 0.0
            p, rc = inter / len(c), inter / len(r)
            return 2 * p * rc / (p + rc)

        def _soft_f1(ce, re_):
            sims = ce @ re_.T
            p  = float(sims.max(dim=1).values.mean())
            rc = float(sims.max(dim=0).values.mean())
            return (2 * p * rc / (p + rc)) if (p + rc) > 0 else 0.0

        want_hard = args.metric in ('all', 'set_match', 'spice')
        want_soft = args.metric in ('all', 'soft_spice')
        encoder = SentenceTransformer(args.text_encoder, device=device) if want_soft else None

        os.makedirs(args.sg_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(args.sg_dir, "*.json")))
        if not files:
            raise SystemExit(f"No .json files found in {args.sg_dir}")

        summary_lines = ["# SPICE / Soft-SPICE summary", "",
                         "| file | n | mean_spice | mean_soft_spice |",
                         "|------|---|------------|-----------------|"]
        all_results = []

        for fp in tqdm.tqdm(files, desc="SPICE"):
            name = os.path.basename(fp)
            if name.startswith('spice_summary'):
                continue
            try:
                samples = json.load(open(fp, "r", encoding='utf-8'))
            except Exception as e:
                print(f"  skip {name}: {e}")
                continue
            if not isinstance(samples, list):
                samples = [samples]

            hard_scores, soft_scores = [], []
            for sample in samples:
                cand_raw = sample.get("unique_sg") or []
                if isinstance(cand_raw, set):
                    cand_raw = list(cand_raw)
                ref_raw = get_anno_sg(_norm_sample_for_get_anno_sg(sample))

                cand = [t for t in (_norm_triple(s) for s in cand_raw) if t]
                ref  = [t for t in (_norm_triple(s) for s in ref_raw)  if t]

                if want_hard:
                    hard_scores.append(_hard_f1(set(cand), set(ref)))

                if want_soft:
                    if cand and ref:
                        with torch.no_grad():
                            ce  = encoder.encode(cand, convert_to_tensor=True,
                                                 normalize_embeddings=True, show_progress_bar=False)
                            re_ = encoder.encode(ref, convert_to_tensor=True,
                                                 normalize_embeddings=True, show_progress_bar=False)
                        soft_scores.append(_soft_f1(ce, re_))
                    else:
                        soft_scores.append(0.0)

            n = len(samples)
            mean_s  = (sum(hard_scores) / len(hard_scores)) if hard_scores else None
            mean_ss = (sum(soft_scores) / len(soft_scores)) if soft_scores else None
            s_str  = f"{mean_s:.4f}"  if mean_s  is not None else "-"
            ss_str = f"{mean_ss:.4f}" if mean_ss is not None else "-"
            summary_lines.append(f"| {name} | {n} | {s_str} | {ss_str} |")
            all_results.append({"file": name, "n": n,
                                "mean_spice":      (float(mean_s)  if mean_s  is not None else None),
                                "mean_soft_spice": (float(mean_ss) if mean_ss is not None else None)})

        valid = [r for r in all_results if r.get("n", 0) > 0]
        if valid:
            total_n = sum(r["n"] for r in valid)
            parts = []
            if want_hard:
                hv = [r for r in valid if r.get("mean_spice") is not None]
                if hv:
                    parts.append(f"SPICE={sum(r['mean_spice']*r['n'] for r in hv)/sum(r['n'] for r in hv):.4f}")
            if want_soft:
                sv = [r for r in valid if r.get("mean_soft_spice") is not None]
                if sv:
                    parts.append(f"Soft-SPICE={sum(r['mean_soft_spice']*r['n'] for r in sv)/sum(r['n'] for r in sv):.4f}")
            if parts:
                summary_lines.append("")
                summary_lines.append(f"**overall** (n={total_n}): " + ", ".join(parts))

        summary_text = "\n".join(summary_lines)
        print(summary_text)
        out_summary = os.path.join(args.sg_dir, "spice_summary.txt")
        with open(out_summary, "w", encoding='utf-8') as f:
            f.write(summary_text)
        print(f"Summary written: {out_summary}")
        with open(os.path.join(args.sg_dir, "spice_summary.json"), "w", encoding='utf-8') as f:
            json.dump({"results": all_results, "summary": summary_text}, f, indent=2, ensure_ascii=False)
        raise SystemExit(0)

    # # 单文件模式（原逻辑）
    # from factual_scene_graph.evaluation.evaluator import Evaluator
    # from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
    # agent = LLMChat("gpt-5")
    # os.makedirs(args.outdir, exist_ok=True)
    # output_path = os.path.join(args.outdir, "output/five_context/InternVL3-8B-Instruct_merged_internal_sg.json")
    # samples = json.load(open(args.conv_script, "r"))
    # print("Start parsing...")
    # anno_conv_list, ref_sg_list = _collect_sg_inputs(samples, agent=agent, sample_num=args.sample_num, do_parse=True)
    # samples = samples[:args.sample_num] if args.sample_num is not None else samples

    # sg_parser = SceneGraphParser(args.sg_model, device=device)
    # evaluator = Evaluator(parser=sg_parser, text_encoder_checkpoint=args.text_encoder, device=device, lemmatize=False)
    # spice_scores, cand_graphs, ref_graphs = evaluator.evaluate(
    #     anno_conv_list, ref_sg_list, method='spice',
    #     beam_size=args.beam_size, batch_size=128, max_input_len=10000, max_output_len=256, return_graphs=True
    # )
    # print('SPICE scores:', sum(spice_scores) / len(spice_scores))
    # if args.metric == 'all':
    #     soft_spice_scores = evaluator.evaluate(cand_graphs, ref_graphs, method='soft_spice', beam_size=1)
    #     print('Soft-SPICE scores:', sum(soft_spice_scores) / len(soft_spice_scores))
    # else:
    #     soft_spice_scores = evaluator.evaluate(cand_graphs, ref_graphs, method=args.metric, beam_size=1)
    #     print(f'{args.metric} scores:', sum(soft_spice_scores) / len(soft_spice_scores))
    # for i in range(len(samples)):
    #     samples[i]["spice"] = spice_scores[i]
    #     samples[i]["soft_spice"] = soft_spice_scores[i]
    # with open(output_path, "w") as f:
    #     json.dump(samples, f, indent=4)
