from openai import OpenAI
import argparse
import json
import time
import requests
from tqdm import tqdm
from utils.coco import format_case_coco
from utils.vg import format_case_vg

from dotenv import load_dotenv
import os

load_dotenv(".env")

# just needs a simple list of objects
# For COCO datasets: Extracts categories from instances field
# For VG datasets: Extracts object names from sg.objects field
# Output format: "cat, dog, person, car" (comma-separated unique objects)

def format_image_content_simple(metadata):
    """Format image content as simple object list for MMHal evaluation."""
    # Extract object categories
    objects = []
    
    # Try different metadata formats
    if 'instances' in metadata:
        # COCO format
        objects = [ins['category'] for ins in metadata['instances']]
    elif 'objects' in metadata:
        # VG format with direct objects array
        for obj_data in metadata['objects']:
            if 'names' in obj_data and obj_data['names']:
                objects.extend(obj_data['names'])
    elif 'sg' in metadata and 'objects' in metadata['sg']:
        # VG format with sg.objects dict
        for obj_id, obj_data in metadata['sg']['objects'].items():
            if 'names' in obj_data and obj_data['names']:
                objects.extend(obj_data['names'])
    
    # Remove duplicates and join
    objects = list(set(objects))
    return ", ".join(objects) if objects else "No objects detected"

template = '''Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.

For clarity, consider these examples:

### Example 1:
- Image Contents: Lynx, Cat
- Question: How many cats can you see in this image?
- Standard Human-Generated Answer: There are two cats visible in the image.
- LMM Response to Evaluate: There is one cat in the image.
- Analysis: The LMM's response is false because it claims there is only one cat in the image, when in fact there are two.
- Hallucination: Yes.

### Example 2:
- Image Contents: Tree, Plant, Panda, Animal
- Question: What can you see from this photo?
- Standard Human-Generated Answer: In this photo we can see branches. We can see a panda on a branch. The background is blurry.
- LMM Response to Evaluate: In the photo, I can see a large black and white panda sitting on a tree branch. The image captures the panda's unique appearance and its natural habitat, showcasing the beauty and rarity of these animals.
- Analysis: The LMM's response is not false because it does not make any false claims about the image contents. For example, it does not claim that the panda is eating bamboo.
- Hallucination: No.

### Example 3:
- Image Contents: Fashion accessory, Mammal, Dog, Umbrella
- Question: Describe this image in details.
- Standard Human-Generated Answer: In this picture I can see a dog laying under a yellow umbrella on a beach, and there is a rope on the right side. In the background I can see few persons talking or lying. At the top of the image, I can see the blue and clear sky.
- LMM Response to Evaluate: The image features a brown dog laying on the sand at the beach, underneath a yellow umbrella. The dog is positioned near the water, and there are people in the background. The beach is sandy, and the sun is shining, creating a pleasant atmosphere. The dog is enjoying the shade provided by the umbrella, which is a common way to protect pets from the sun's harmful rays. The presence of people in the background suggests that the beach is a popular spot for leisure activities, such as sunbathing, swimming, and relaxing.
- Analysis: The LMM's response is false because it claims that the dog is positioned near the water, but there is no water in the image.
- Hallucination: Yes.

### Example 4:
- Image Contents: Clothing, Luggage and bags, Man, Building, Woman, Person
- Question: Where is this photo taken?
- Standard Human-Generated Answer: From the slogan "Welcome to Houston, Texas", we can infer that this photo is taken at an airport in Houston, Texas.
- LMM Response to Evaluate: This photo is taken at an airport, specifically at the entrance of a terminal. However, the specific location of this airport cannot be determined from the image.
- Analysis: Although the LMM's response does not explicitly mention the location "Houston, Texas" and does not perfectly answer the question, it makes no false claims about the image contents. Therefore, this response is not considered hallucination. It is OK if the LMM fails to directly answer the question, expresses uncertainty, or cannot include all the objects or details in the human-generated answer.
- Hallucination: No.

### Example 5:
- Image Contents: Tree, Tower
- Question: What environmental concerns are involved in this photo?
- Standard Human-Generated Answer: This photo shows a smokestack emitting smoke. It may cause severe air pollution which threatens human health.
- LMM Response to Evaluate: In the photo, there are several environmental concerns related to the smokestack emitting smoke. The smoke from the smokestack is a byproduct of industrial processes, which can contribute to air pollution, climate change, and negative impacts on human health. The smoke contains harmful pollutants, such as particulate matter, sulfur dioxide, nitrogen oxides, and carbon monoxide, which can lead to respiratory problems, heart disease, and other health issues. Additionally, the smoke contributes to the greenhouse effect, which can lead to global warming and climate change. The smokestack's emissions also affect the environment, as they can harm wildlife, vegetation, and ecosystems. Therefore, it is essential to address these environmental concerns by implementing measures to reduce emissions and promote sustainable practices in industrial processes.
- Analysis: Although the LMM's response is significantly longer than the standard human-generated answer, it does not contain any false claims about the image contents. Instead, it provides additional general information about the environmental concerns, which can be inferred from the smoke emission. Such detailed analysis or reasoning should be considered as a positive aspect, as long as it contains no false claims.
- Hallucination: No.

With these examples in mind, please help me evaluate whether the response by the LMM is informative, and whether hallucination exists in it, based on the comparison between the LMM's response and the factual information provided in the image contents, question, and the standard human-generated answer below.

Please note that the standard human-generated answer may only contain factual information but may not give a detailed analysis. Also, the standard human-generated answer may not be completely comprehensive in describing all the objects and their attributes, so please be a bit more cautious during evalutation. LMM's detailed analysis or reasoning should be encouraged.

To evaluate the LMM responses, first, begin your evaluation by providing a short explanation. Second, after providing your explanation, you must rate the response by choosing from the following options:
- Rating: 6, very informative with good analysis or reasoning, no hallucination
- Rating: 5, very informative, no hallucination
- Rating: 4, somewhat informative, no hallucination
- Rating: 3, not informative, no hallucination
- Rating: 2, very informative, with hallucination
- Rating: 1, somewhat informative, with hallucination
- Rating: 0, not informative, with hallucination

### Image Contents
{}

### Question
{}

### Standard Human-Generated Answer
{}

### LMM Response to Evaluate
{}
'''

score_parsing_template = '''Extract the numerical rating from the following evaluation text. 
Return ONLY a single digit from 0-6, nothing else.

Evaluation text:
{}
'''


def parse_score_with_llm(api_url, api_key, evaluation_text, gpt_model, max_retries=5, timeout=120):
    """Use LLM to parse the score from evaluation text."""
    for attempt in range(max_retries):
        try:
            headers = {
                "Content-Type": "application/json"
            }
            # Only add Authorization header if API key is provided
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            payload = {
                "model": gpt_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that extracts numerical ratings from text. Return only a single digit."},
                    {"role": "user", "content": score_parsing_template.format(evaluation_text)}
                ],
                "temperature": 0.0,
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "0").strip()
            
            try:
                score = int(content)
                if 0 <= score <= 6:
                    return score
                else:
                    print(f"Warning: Score {score} out of range, defaulting to 0")
                    return 0
            except ValueError:
                print(f"Warning: Could not parse score from: {content}")
                return 0
        except Exception as e:
            print(f"Error parsing score (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print('retrying...')
                time.sleep(10)
            else:
                print(f"Failed to parse score after {max_retries} attempts")
                return 0  # Default score on failure
    
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--response', type=str, default='output/dyna_bad_examples/coverage_certainty_with_answer_json_mode.json', help='response file containing images, questions, and model responses')
    parser.add_argument('--evaluation', type=str, default=None, help='GPT-4 evaluation results to be saved')
    parser.add_argument('--gpt-model', type=str, default='gpt-4o-mini')
    parser.add_argument('--gt-type', type=str, default='dyna', choices=['dyna', 'caption'], 
                        help='Type of ground truth to use: "dyna" uses gt field from conversations, "caption" generates from metadata')
    parser.add_argument('--api-url', type=str, default=None,
                       help='API endpoint URL (e.g., https://.../v1/chat/completions). Overrides OPENAI_BASE_URL env var.')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API authorization key. Overrides OPENAI_API_KEY env var.')
    args = parser.parse_args()

    # Set default evaluation path if not provided
    if args.evaluation is None:
        base_name = os.path.splitext(args.response)[0]
        args.evaluation = f"{base_name}_evaluation.json"

    # Get API credentials - use args if provided, otherwise fall back to env vars
    api_key = args.api_key if args.api_key else os.getenv("OPENAI_API_KEY")
    api_url = args.api_url if args.api_url else os.getenv("OPENAI_BASE_URL")
    
    # If api_url doesn't end with /chat/completions, add it (for requests.post)
    if api_url and not api_url.endswith('/chat/completions'):
        if api_url.endswith('/'):
            api_url = api_url + 'chat/completions'
        else:
            api_url = api_url + '/chat/completions'

    # load json file
    with open(args.response, 'r') as f:
        records = json.load(f)

    # Check if evaluation already exists
    if os.path.exists(args.evaluation):
        print(f"Loading existing evaluation from {args.evaluation}")
        with open(args.evaluation, 'r') as f:
            evaluation_data = json.load(f)
            detailed_results = evaluation_data.get('detailed_results', [])
    else:
        # ask GPT-4 to evaluate
        detailed_results = []
        conv_index = 0
        
        for i, record in enumerate(tqdm(records, desc="Evaluating")):
            # Get metadata and format as simple object list
            if 'metadata' in record:
                image_content = format_image_content_simple(record['metadata'])
            else:
                # Assume record is already in COCO/VG format
                image_content = format_image_content_simple(record)
            
            for one_round_conv in record['conversations']:
                # Get ground truth based on gt-type argument
                if args.gt_type == 'dyna':
                    # Use gt field from conversation (for dyna_conv outputs)
                    gt_answer = one_round_conv['gt']
                else:  # args.gt_type == 'caption'
                    # Generate from metadata using format_case_vg
                    gt_answer = format_case_vg(record, use_region=True)
                
                input_text = template.format(
                    image_content, 
                    one_round_conv['prompt'], 
                    gt_answer,
                    one_round_conv['response']
                )

                max_retries = 5
                timeout = 120
                evaluation_text = None
                
                for attempt in range(max_retries):
                    try:
                        headers = {
                            "Content-Type": "application/json"
                        }
                        # Only add Authorization header if API key is provided
                        if api_key:
                            headers["Authorization"] = f"Bearer {api_key}"
                        
                        payload = {
                            "model": args.gpt_model,
                            "messages": [
                                {"role": "system", "content": "You are a helpful, impartial and objective judge that can accurately evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question."},
                                {"role": "user", "content": input_text}
                            ],
                            "temperature": 0.0,
                        }
                        
                        response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
                        response.raise_for_status()
                        
                        result = response.json()
                        evaluation_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        break
                    except Exception as e:
                        print(f"Error in evaluation (attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            print('retrying...')
                            time.sleep(10)
                        else:
                            print(f"Failed to get evaluation after {max_retries} attempts, skipping this item")
                            break  # Break out of retry loop

                if evaluation_text is None:
                    continue  # Skip if all retries failed

                # Parse score using LLM
                score = parse_score_with_llm(api_url, api_key, evaluation_text, args.gpt_model, max_retries=max_retries, timeout=timeout)
                
                # Store detailed result
                detailed_results.append({
                    'record_index': i,
                    'round_id': one_round_conv.get('round_id', conv_index),
                    'q_type': one_round_conv.get('q_type', 'unknown'),
                    'prompt': one_round_conv['prompt'],
                    'response': one_round_conv['response'],
                    'gt': gt_answer,
                    'evaluation': evaluation_text,
                    'score': score,
                    'has_hallucination': score < 3
                })
                
                if conv_index % 5 == 0:
                    print(f"{conv_index} | Score: {score} | {evaluation_text[:100]}...", flush=True)
                
                conv_index += 1
                time.sleep(0.1)

    # Calculate aggregated metrics
    scores = [r['score'] for r in detailed_results]
    hallucinations = [r['has_hallucination'] for r in detailed_results]
    
    # Overall metrics
    overall_metrics = {
        'avg_score': sum(scores) / len(scores) if scores else 0,
        'hallucination_rate': sum(hallucinations) / len(hallucinations) if hallucinations else 0,
        'total_evaluations': len(detailed_results)
    }
    
    # Metrics by q_type
    q_type_metrics = {}
    q_types = set(r['q_type'] for r in detailed_results)
    
    for q_type in q_types:
        q_type_results = [r for r in detailed_results if r['q_type'] == q_type]
        q_type_scores = [r['score'] for r in q_type_results]
        q_type_hallucinations = [r['has_hallucination'] for r in q_type_results]
        
        q_type_metrics[q_type] = {
            'avg_score': sum(q_type_scores) / len(q_type_scores) if q_type_scores else 0,
            'hallucination_rate': sum(q_type_hallucinations) / len(q_type_hallucinations) if q_type_hallucinations else 0,
            'count': len(q_type_results)
        }
    
    # Prepare output
    output_data = {
        'overall_metrics': overall_metrics,
        'metrics_by_q_type': q_type_metrics,
        'detailed_results': detailed_results
    }
    
    # Save results
    os.makedirs(os.path.dirname(args.evaluation) if os.path.dirname(args.evaluation) else '.', exist_ok=True)
    with open(args.evaluation, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\nOverall Metrics:")
    print(f"  Average Score: {overall_metrics['avg_score']:.3f}")
    print(f"  Hallucination Rate: {overall_metrics['hallucination_rate']:.3f}")
    print(f"  Total Evaluations: {overall_metrics['total_evaluations']}")
    
    print(f"\nMetrics by Question Type:")
    for q_type, metrics in sorted(q_type_metrics.items()):
        print(f"  {q_type}:")
        print(f"    Average Score: {metrics['avg_score']:.3f}")
        print(f"    Hallucination Rate: {metrics['hallucination_rate']:.3f}")
        print(f"    Count: {metrics['count']}")
    
    print(f"\nResults saved to: {args.evaluation}")
    print("="*80)