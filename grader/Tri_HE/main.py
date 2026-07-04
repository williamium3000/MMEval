import argparse
import json
import os
import copy
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

from graders.Tri_HE.extractor import GPT4Extractor


def load_ref_triplet(sample):
    ref_triplets = []
    for attr_object in sample["attributes"]:
        if attr_object["attributes"] is None or len(attr_object["attributes"]) == 0:
            ref_triplets += [f"{attr_object['names'][0]}"]
        else:
            for attr in attr_object["attributes"]:
                ref_triplets += [f"{attr_object['names'][0]}, is, {attr}"]

    for rel in sample['relationships']:
        ref_triplets += [f"{rel['subject']['names'][0]}, {rel['predicate'].lower()}, {rel['object']['names'][0]}"]
    return list(set(ref_triplets))


def gpt_judge(judge, judge_model, ref_triplets, object_list, t):
    evaluate_instruction = 'Given a list of reference triplets ("object1", "relation", "object2") extracted from the scene graph of an image, along with a list of objects observed in this image, your task is:\n\n' \
                           'Task 1. Determine if a claim triplet ("object1", "relation", "object2") is directly supported by any single triplet in the reference, or can be logically inferred from multiple reference triplets and the list of objects. Follow these steps when finishing the task:\n\n' \
                           '1. Answer "yes" if the claim appears in the reference.\n\n' \
                           '2. Answer "yes" if the claim can be logically inferred from one or more triplets in the reference. Consider:\n\n' \
                           'a. General Inferences: Assess common associations or implications.\n' \
                           'b. Conditional Phrases: Note phrases like "could be", "might", "suggests", which allow broader inferences.\n' \
                           'c. Equivalence of Objects: In your judgment, treat objects of the same kind as equal. For example, "woman", "man" should be considered under the general category of "person".\n' \
                           'd. Support from Object List: If the claim is not directly supported or inferable from the triplets, assess whether the list of objects provides additional evidence to support or infer the claim.\n\n' \
                           '3. Answer "no" if the claim neither directly matches any triplet in the reference nor can be reasonably inferred from the triplets and the object list.\n\n' \
                           'Task 2: Error categorization.\n\n' \
                           'If your answer to the previous task is "no", determine whether the not supported/inferred part in the claim is "object1" or "object2" or "relation".\n\n' \
                           'Reference:\n{}\n\n' \
                           'List of Objects:\n{}\n\n' \
                           'Claim:\n{}\n\n' \
                           'Please output your answer to the first task only in the format of "My answer is \'yes\'/\'no\'". If your answer is "no", output your answer to the second task only in the format of "The error is related to \'object1\'/\'object2\'/\'relation\'".'

    ref_list = list(set([tuple(item.split(', ')) for item in ref_triplets]))
    judge_prompt = copy.deepcopy(evaluate_instruction).format(ref_list, object_list, tuple(t), '{}', '{}', '{}')

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": judge_prompt}]

    response = judge.chat.completions.create(
        model=judge_model,
        messages=messages,
        # temperature=0.7,
    )

    response = response.choices[0].message.content

    if ("my answer is 'yes'" in response.lower()) or (
            "my answer is \"yes\"" in response.lower()):
        return 'yes', response
    elif ("my answer is 'no'" in response.lower()) or (
            "my answer is \"no\"" in response.lower()):
        return 'no', response
    else:
        return 'null', response


def NLI_judge(judge, ref_triplets, t):
    embeddings = judge.encode(ref_triplets)
    src = judge.encode([' '.join(t)])
    res = cosine_similarity(src, embeddings)
    # filter not useful triplets
    filtered = [' '.join([ref_triplets[idx] for idx in np.nonzero(res > 0.5)[1]])]
    topk = (-res).argsort()[0, :3]
    oriembeddings = judge.encode([' '.join([ref_triplets[tdx] for tdx in topk])])
    if filtered == ['']:
        # print('filtered', [ref_triplets[tdx] for tdx in topk])
        entailval = cosine_similarity(src, oriembeddings)[0][0]
    else:
        # print('filtered', filtered)
        orival = cosine_similarity(src, oriembeddings)[0][0]
        newembeddings = judge.encode(filtered)
        entailval = cosine_similarity(src, newembeddings)[0][0]
        entailval = max(entailval, orival)

    if entailval > 0.6:
        return 'yes'
    else:
        return 'no'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv', type=str, default='output/vg/icl.json')
    parser.add_argument('--outfile', type=str, default='output/vg/icl_trihe.json')
    parser.add_argument('--extractor', type=str, default='gpt-4o-mini')
    parser.add_argument('--judge', type=str, default='NLI', choices=['gpt', 'NLI'])
    parser.add_argument('--judge_model', type=str, default='gpt-4o-mini')
    args = parser.parse_args()

    conv_data = json.load(open(args.conv, 'r'))
    extractor = GPT4Extractor()
    if args.judge == 'NLI':
        judge = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    elif args.judge == 'gpt':
        judge = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise NotImplementedError("This judge is not yet implemented.")

    lsofres1 = []
    lvlm_score = []
    lvlm_scores = []
    lvlm_object_scores = []
    lvlm_relation_scores = []
    pbar = tqdm(conv_data)
    for sample in tqdm(conv_data):
        objects = 0
        relations = 0
        lvlm_object_score = []
        lvlm_relation_score = []
        ref_triplets = load_ref_triplet(sample)
        for conv in sample["conversations"]:
            if 'triplet' not in conv.keys():
                response = conv["response"].strip().replace('\n', '')
                triplets = extractor.extract(response, model=args.extractor)
                conv['triplet'] = triplets
            else:
                triplets = conv['triplet']

            lsofres1.append([])
            judgements = []
            for tdx, t in enumerate(triplets):
                # print('evaluated_triplet', t)
                if args.judge == 'NLI':
                    ref_triplets = [ref.replace(',', '') for ref in ref_triplets]
                    judge_res = NLI_judge(judge, ref_triplets, t)
                elif args.judge == 'gpt':
                    object_list = set([object_dict["names"][0] for object_dict in sample["objects"]])
                    judge_res, judgement = gpt_judge(judge, args.judge_model, ref_triplets, list(object_list), t)
                    judgements.append(judgement)
                    if ("my answer is 'no'" in judgement.lower()) or ("my answer is \"no\"" in judgement.lower()):
                        if ("the error is related to 'object1'" in judgement.lower()) or ("the error is related to 'object2'" in judgement.lower()):
                            objects += 1
                        else:
                            relations += 1
                else:
                    raise NotImplementedError("This function is not yet implemented.")

                lsofres1[-1].append(judge_res)

            conv[f'{args.judge}_nli'] = lsofres1[-1]
            try:
                lvlm_score.append(lsofres1[-1].count('no') / len(lsofres1[-1]))
                if args.judge == 'gpt':
                    lvlm_object_score.append(objects / len(judgements))
                    lvlm_relation_score.append(relations / len(judgements))
            except ZeroDivisionError:
                continue

        # sample level
        lsofresall = []
        for l in lsofres1:
            lsofresall += l
        yes_count, no_count = len([r for r in lsofresall if r == 'yes']), len([r for r in lsofresall if r == 'no'])
        pbar.set_postfix({'entail': yes_count / (yes_count + no_count)})
        lvlm_scores.append(sum(lvlm_score) / len(lvlm_score))

        if args.judge == 'gpt':
            lvlm_object_scores.append(sum(lvlm_object_score) / len(lvlm_object_score))
            lvlm_relation_scores.append(sum(lvlm_relation_score) / len(lvlm_relation_score))

    print("Hallu-I: ", sum(lvlm_scores) / len(lvlm_scores))
    print("Hallu-Q: ", sum(lvlm_score) / len(lvlm_score))

    if args.judge == 'gpt':
        print('Object Hallu-I: ', sum(lvlm_object_scores) / len(lvlm_object_scores))
        print('Relation Hallu-I: ', sum(lvlm_relation_scores) / len(lvlm_relation_scores))

    with open(args.outfile, 'w') as file:
        json.dump(conv_data, file)