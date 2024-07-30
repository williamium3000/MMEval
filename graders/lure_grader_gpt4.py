import openai
import argparse
import json
import time

template = '''Suppose you are a hallucination annotator who judges the degree of hallucination based on objects, and you have the following image information. 
Reference captions:{five captions from COCO} 
Bounding box:{bounding boxes} 
Please just provide the ranks for the below descriptions without any explanation, where the caption ranks first with the most hallucinations. 
The output format: [caption x,...] 
Descriptions: 
caption 1: {description 1} 
caption 2: {description 2} 
caption 3: {description 3} 
caption 4: {description 4} 
caption 5: {description 5} 
Output:
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--response', type=str, default='responses/idefics_80b.json', help='response file containing images, questions, and model responses')
    parser.add_argument('--evaluation', type=str, default=None, help='GPT-4 evaluation results to be saved')
    parser.add_argument('--api-key', type=str, required=True)
    parser.add_argument('--gpt-model', type=str, default='gpt-4-0314')
    args = parser.parse_args()

    openai.api_key = args.api_key

    # load json file
    with open(args.response, 'r') as f:
        records = json.load(f)

    assert len(records) == 96

    # ask GPT-4 to evaluate
    responses = []
    for i, record in enumerate(records):
        image_content = ', '.join(record['image_content'])
        input_text = template.format(image_content, record['question'], record['gt_answer'], record['model_answer'])
        # print(input_text)

        response = None
        while response is None:
            try:
                response = openai.ChatCompletion.create(
                    model=args.gpt_model,
                    messages=[
                        {"role": "user", "content": input_text}
                    ],
                    temperature=0.0,
                )
            except Exception as e:
                print(e)
                print('retrying...')
                time.sleep(10)
                continue

        print(i, response['choices'][0]['message']['content'], flush=True)
        responses.append(response)
        time.sleep(1)

    # save responses
    if args.evaluation is not None:
        with open(args.evaluation, 'w') as f:
            json.dump(responses, f, indent=2)

    # analyze responses
    scores = []
    for i, response in enumerate(responses):
        response = response['choices'][0]['message']['content']
        scores_found = []
        for s in range(7):
            if f'rating: {s}' in response.lower():
                scores_found.append(s)
        if len(scores_found) == 1:
            scores.append(scores_found[0])
        else:
            print('Warning: multiple or zero scores found')
            print(i, response)
            scores.append(0)

    hallucination = []
    for s in scores:
        if s >= 3:
            hallucination.append(0)
        else:
            hallucination.append(1)

    scores_each = [[] for _ in range(8)]
    # assuming order of 96 questions is not changed
    for i in range(96):
        question_type = i % 8
        scores_each[question_type].append(scores[i])

    print('Average score: {:.2f}'.format(sum(scores) / len(scores)))
    print('Hallucination rate: {:.2f}'.format(sum(hallucination) / len(hallucination)))
    print('Average score for each question type:', ','.join([str(round(sum(scores_each[i]) / len(scores_each[i]), 2)) for i in range(8)]), flush=True)
