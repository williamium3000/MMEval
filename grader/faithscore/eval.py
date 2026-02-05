import argparse
import json
import os
from tqdm import tqdm
from framework import FaithScore, merge_faithscore_results, recalculate_from_judgments
from io import BytesIO
import requests

# import nltk
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'merge', 'recalculate'])
    
    # eval
    parser.add_argument('--conv', type=str, default='output/vg/caption.json')
    parser.add_argument('--vem_type', type=str, default="llava", choices=["ofa-ve", "ofa", "llava"])
    parser.add_argument('--llava_path', type=str, default="checkpoints/llava-v1.5-7b")
    parser.add_argument('--openai_model', type=str, default='gpt-5')
    parser.add_argument('--use_llama', action='store_true')
    parser.add_argument('--llama_path', type=str)
    parser.add_argument('--sample_num', type=int, default=100)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--save_judgments', type=str, default=None, 
                        help='Path to save LLM judgment results (JSON file)')
    
    # merge
    parser.add_argument('--judgment_files', type=str, nargs='+', default=None,
                        help='file to be merged')
    parser.add_argument('--merge_output', type=str, default=None,
                        help='merge output path')
    
    # recalculate
    parser.add_argument('--judgment_file', type=str, default=None,
                        help='file needs to be recalculated')
    
    args = parser.parse_args()

    if args.mode == 'merge':
        # merge results
        if not args.judgment_files:
            print("Error: --judgment_files is required for merge mode")
            exit(1)
        merge_faithscore_results(args.judgment_files, args.merge_output)
    
    elif args.mode == 'recalculate':
        # recalculate results
        if not args.judgment_file:
            print("Error: --judgment_file is required for recalculate mode")
            exit(1)
        recalculate_from_judgments(args.judgment_file)
    
    else:  # eval
        api_key = os.getenv("OPENAI_API_KEY")
        scorer = FaithScore(vem_type=args.vem_type, api_key=api_key, openai_model=args.openai_model,
                           llava_path=args.llava_path, use_llama=args.use_llama,
                           llama_path=args.llama_path)

        images = []
        answers = []
        conv_data = json.load(open(args.conv, 'r'))

        end_idx = args.start_idx + args.sample_num
        selected_data = conv_data[args.start_idx:end_idx]
        
        print(f"Processing samples {args.start_idx} to {min(end_idx, len(conv_data))-1}")
        
        for sample in tqdm(selected_data):
            for conv in sample["conversations"]:
                response = conv["response"].strip().replace('\n', '')
                image_url = sample["url"]

                try:
                    request_response = requests.get(image_url)
                    images.append(BytesIO(request_response.content))
                    answers.append(response)
                except requests.exceptions.RequestException as e:
                    print(f"Error retrieving image: {e}")
                    continue

        score, sentence_score = scorer.faithscore(answers, images, save_judgments_path=args.save_judgments)

        print('Overall score:', score)
        print('Sentence score:', sentence_score)
