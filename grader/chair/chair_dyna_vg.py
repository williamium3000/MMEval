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
import requests

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


def detect_negative_mentions_via_api(conversation_text, object_list, api_url, api_key, model="Qwen3-30B-A3B-Instruct-2507", max_retries=5, timeout=120):
    """
    Use API to detect which objects are mentioned negatively in the conversation.
    
    Args:
        conversation_text: Full concatenated conversation (all prompts + responses)
        object_list: List of objects/nodes to check for negative mentions
        api_url: API endpoint URL
        api_key: API authorization key
        max_retries: Maximum number of retry attempts (default: 5)
        timeout: Request timeout in seconds (default: 120)
    
    Returns:
        Tuple of (list of objects that are negatively mentioned, error_flag)
        error_flag is True if all retries failed, False otherwise
    """
    for attempt in range(max_retries):
        try:
            # Prepare the prompt for the API
            objects_str = ", ".join([str(obj) for obj in object_list])
            prompt = f"""Given the following conversation and a list of objects, determine which objects from the list are mentioned NEGATIVELY in the conversation.

A negative mention means the object is stated as NOT present, absent, missing, or unavailable. Examples:
- "there is no X"
- "I don't see X"
- "without X"
- "X is not present"
- "there isn't any X"
- "no X visible"

Conversation:
{conversation_text}

Objects to check (only return objects from this list):
{objects_str}

For each object in the list above, determine if it is mentioned negatively in the conversation. Return ONLY the objects from the provided list that are negatively mentioned.

Output format (JSON only):
```json
{{
  "negative_mentioned_objects": ["object1", "object2", ...]
}}
```

Important:
- Only include objects from the provided list
- Only include objects that are mentioned NEGATIVELY (not present/absent)
- Do NOT include objects that are mentioned positively or neutrally
- Return an empty list if no objects are mentioned negatively"""
            
            headers = {
                "Content-Type": "application/json"
            }
            # Only add Authorization header if API key is provided
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # Parse JSON from response
            # Try to extract JSON if it's wrapped in markdown code blocks
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end == -1:
                    json_end = len(content)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                if json_end == -1:
                    json_end = len(content)
                content = content[json_start:json_end].strip()
            
            # Try to find JSON object in the content
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Try to find JSON object pattern
                import re
                json_match = re.search(r'\{[^{}]*"negative_mentioned_objects"[^{}]*\}', content)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    if attempt == max_retries - 1:
                        print(f"Warning: Could not parse API response as JSON after {max_retries} attempts: {content[:200]}")
                        return [], True
                    continue  # Retry
            
            negative_objects = parsed.get("negative_mentioned_objects", [])
            
            # Ensure it's a list
            if not isinstance(negative_objects, list):
                negative_objects = []
            
            return negative_objects, False  # Success, no error
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"API timeout (attempt {attempt + 1}/{max_retries}), retrying...")
                continue
            else:
                print(f"API timeout after {max_retries} attempts")
                return [], True  # Error after all retries
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                continue
            else:
                print(f"API call failed after {max_retries} attempts: {e}")
                return [], True  # Error after all retries
    
    # Should not reach here, but just in case
    return [], True


def extract_negated_objects_from_api(cap_eval, total_synsets, api_url=None, api_key=None, model="Qwen3-30B-A3B-Instruct-2507"):
    """
    Extract negated objects using API call.
    Returns a set of negated object words (stemmed) and stores in negative_mentioned_objects key.
    """
    negated_words = set()
    
    # Build full conversation text (all prompts + responses)
    conversation_parts = []
    for conv in cap_eval.get('conversations', []):
        if conv.get('prompt'):
            conversation_parts.append(f"Question: {conv['prompt']}")
        if conv.get('response'):
            conversation_parts.append(f"Answer: {conv['response']}")
    
    conversation_text = "\n".join(conversation_parts)
    
    # If no conversations, use caption as fallback
    if not conversation_text:
        conversation_text = cap_eval.get('caption', '')
    
    if not conversation_text:
        cap_eval['negative_mentioned_objects'] = []
        return negated_words
    
    # Extract all potential objects from the conversation text first
    raw_words = sentence2words(conversation_text)
    matched_objects = match_vg_list(total_synsets, raw_words)
    object_list = [obj[0] for obj in matched_objects]  # Get object names
    
    # Call API if available (api_key can be empty for local vLLM)
    if api_url and object_list:
        negative_objects, api_error = detect_negative_mentions_via_api(
            conversation_text, 
            object_list, 
            api_url, 
            api_key,
            model
        )
        
        # Mark sample as error if API failed
        if api_error:
            cap_eval['api_error'] = True
            cap_eval['negative_mentioned_objects'] = []
        else:
            # Store in the sample (use list to avoid overwriting if called multiple times)
            if 'negative_mentioned_objects' not in cap_eval:
                cap_eval['negative_mentioned_objects'] = []
            cap_eval['negative_mentioned_objects'].extend(negative_objects)
            cap_eval['negative_mentioned_objects'] = list(set(cap_eval['negative_mentioned_objects']))
            
            # Convert to stemmed set for filtering
            for obj in negative_objects:
                stemmed = ps.stem(obj.lower())
                negated_words.add(stemmed)
                negated_words.add(obj.lower())
    else:
        # Fallback: no API, return empty set
        if 'negative_mentioned_objects' not in cap_eval:
            cap_eval['negative_mentioned_objects'] = []
        cap_eval['api_error'] = False
    
    return negated_words


def extract_gt_synsets(cap_eval):
    """Extract ground truth synsets from a sample."""
    gt_synsets = []
    # objects
    for obj in cap_eval.get('objects', []):
        gt_synsets.extend(obj.get('synsets', []))
    # attributes
    for attr in cap_eval.get('attributes', []):
        gt_synsets.extend(attr.get('synsets', []))
    # relationships
    for rel in cap_eval.get('relationships', []):
        gt_synsets.extend(rel.get("subject", {}).get('synsets', []))
        gt_synsets.extend(rel.get("object", {}).get('synsets', []))
    # regions
    for rcap in cap_eval.get('regions', []):
        if isinstance(rcap, dict):
            phrase = rcap.get('phrase', '')
        else:
            phrase = str(rcap)
        if phrase:
            raw_words_recap = sentence2words(phrase)
            # Note: We can't match to synsets here without the synset dict, 
            # so we'll skip region-based synsets for now in this helper
            # The full extraction will handle this in the compute functions
    return list(set(gt_synsets))


def match_vg_list(object_dict, words, negated_words_set=None):
    """
    Match words to VG objects, excluding negated words.
    
    Args:
        object_dict: Dictionary mapping words to (word, synset) tuples
        words: List of words to match
        negated_words_set: Set of negated words (stemmed) to exclude
    """
    if negated_words_set is None:
        negated_words_set = set()
    
    match_result = []
    for vg_object in object_dict.keys():
        for word in words:
            # Check if this word matches the VG object
            if vg_object == word:
                # Check if this word is negated
                word_stemmed = ps.stem(word.lower())
                if word_stemmed not in negated_words_set and word.lower() not in negated_words_set:
                    match_result.append(vg_object)
    
    match_result = list(set([(object_dict[match][0], object_dict[match][1]) for match in match_result]))
    return match_result

def compute_chair(caps, total_synsets, group_by_image=False, api_url=None, api_key=None, model="Qwen3-30B-A3B-Instruct-2507"):

    '''
    Given ground truth objects and generated captions, determine which sentences have hallucinated words.
    Negative mentions of objects are excluded from matching.
    
    Args:
        caps: List of caption samples
        total_synsets: Dictionary mapping words to synsets
        group_by_image: If True, group items by image_id and use shared GT synsets per image
        api_url: API endpoint URL for negative mention detection (optional)
        api_key: API authorization key (optional)
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

    # Track seen images for coverage when grouping
    seen_images_for_coverage = set()
    
    # Track error samples
    error_samples = []

    # Group by image_id if requested
    if group_by_image:
        from collections import defaultdict
        image_groups = defaultdict(list)
        for cap_eval in caps:
            imid = cap_eval.get('image_id', 'unknown')
            image_groups[imid].append(cap_eval)
        
        # Build shared GT synsets per image_id (merge from all items with same image_id)
        image_gt_synsets = {}
        for imid, group_items in image_groups.items():
            gt_synsets = []
            for cap_eval in group_items:
                # objects
                for obj in cap_eval.get('objects', []):
                    gt_synsets.extend(obj.get('synsets', []))
                # attributes
                for attr in cap_eval.get('attributes', []):
                    gt_synsets.extend(attr.get('synsets', []))
                # relationships
                for rel in cap_eval.get('relationships', []):
                    gt_synsets.extend(rel.get("subject", {}).get('synsets', []))
                    gt_synsets.extend(rel.get("object", {}).get('synsets', []))
                # regions
                for rcap in cap_eval.get('regions', []):
                    if isinstance(rcap, dict):
                        phrase = rcap.get('phrase', '')
                    else:
                        phrase = str(rcap)
                    if phrase:
                        raw_words_recap = sentence2words(phrase)
                        output_objects_recap = match_vg_list(total_synsets, raw_words_recap)
                        synset_recap = [_[1] for _ in output_objects_recap]
                        gt_synsets.extend(synset_recap)
            image_gt_synsets[imid] = list(set(gt_synsets))
        
        # Flatten groups back to list for processing
        caps = [item for group in image_groups.values() for item in group]

    for i, cap_eval in enumerate(tqdm.tqdm(caps)):
        cap = cap_eval['caption']
        imid = cap_eval['image_id']

        # Extract negated objects using API first (this sets api_error flag if it fails)
        # api_key can be empty for local vLLM
        if api_url:
            negated_words = extract_negated_objects_from_api(cap_eval, total_synsets, api_url, api_key, model)
        else:
            # No API available, return empty set (no negative filtering)
            negated_words = set()
            cap_eval['negative_mentioned_objects'] = []
            cap_eval['api_error'] = False
        
        # Skip samples with API errors
        if cap_eval.get('api_error', False):
            error_samples.append(imid)
            continue

        # Use shared GT if grouping, otherwise extract per-item
        if group_by_image:
            gt_synsets = image_gt_synsets.get(imid, [])
        else:
            gt_synsets = []
            # objects
            for obj in cap_eval.get('objects', []):
                gt_synsets.extend(obj.get('synsets', []))
            # attributes
            for attr in cap_eval.get('attributes', []):
                gt_synsets.extend(attr.get('synsets', []))
            # relationships
            for rel in cap_eval.get('relationships', []):
                gt_synsets.extend(rel.get("subject", {}).get('synsets', []))
                gt_synsets.extend(rel.get("object", {}).get('synsets', []))
            # regions
            for rcap in cap_eval.get('regions', []):
                if isinstance(rcap, dict):
                    phrase = rcap.get('phrase', '')
                else:
                    phrase = str(rcap)
                if phrase:
                    raw_words_recap = sentence2words(phrase)
                    output_objects_recap = match_vg_list(total_synsets, raw_words_recap)
                    synset_recap = [_[1] for _ in output_objects_recap]
                    gt_synsets.extend(synset_recap)
            
            gt_synsets = list(set(gt_synsets))
        
        raw_words = sentence2words(cap)
        
        # Match words to synsets, excluding negated words
        output_objects = match_vg_list(total_synsets, raw_words, negated_words_set=negated_words)
        
        
        cap_dict = {'image_id': cap_eval['image_id'], 
                    'caption': cap,
                    'vg_hallucinated_words': [],
                    'vg_gt_words': gt_synsets,
                    'vg_generated_words': [_[0] for _ in output_objects],
                    "words": raw_words,
                    "negative_mentioned_objects": cap_eval.get('negative_mentioned_objects', [])
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
        
        # For coverage: if grouping by image, only add GT once per image
        # Otherwise, add GT for each item
        if group_by_image:
            if imid not in seen_images_for_coverage:
                output['gt_objects'].extend(gt_synsets)
                seen_images_for_coverage.add(imid)
                if len(gt_synsets) > 0:
                    output["coverage"].append(len(list(set(object_in_gts))) / len(gt_synsets))
        else:
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

    chair_s = (num_hallucinated_caps/num_caps) if num_caps > 0 else 0
    chair_i = (hallucinated_word_count/vg_word_count) if vg_word_count > 0 else 0
    chair_i_v2 = len(output['vg_hallucinated_words'])/len(output['vg_words']) if len(output['vg_words']) > 0 else 0
    coverage_avg = np.mean(output["coverage"]) if len(output["coverage"]) > 0 else 0
    coverage_all = len(output['object_in_gts']) / len(output['gt_objects']) if len(output['gt_objects']) > 0 else 0
    output['overall_metrics'] = {
                                    'CHAIRs': chair_s,
                                    'CHAIRi': chair_i,
                                    "CHAIRi_v2": chair_i_v2,
                                    "Coverage_avg": coverage_avg,
                                    "Coverage_all": coverage_all,
                                    "error_samples_count": len(error_samples),
                                    "error_samples": error_samples
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
        
        # Report errors if any
        error_count = sentence_metrics.get('error_samples_count', 0)
        if error_count > 0:
            print(f"\n⚠️  Warning: {error_count} sample(s) had API errors and were excluded from calculations")
            error_samples = sentence_metrics.get('error_samples', [])
            if len(error_samples) <= 10:
                print(f"   Error samples: {error_samples}")
            else:
                print(f"   Error samples (first 10): {error_samples[:10]}...")
                print(f"   (and {len(error_samples) - 10} more)")

    else:
        return metric_string


def compute_chair_by_qtype(caps, total_synsets, group_by_image=False, api_url=None, api_key=None):
    """
    Compute CHAIR metrics grouped by question type.
    Processes each conversation round separately.
    
    Args:
        caps: List of caption samples
        total_synsets: Dictionary mapping words to synsets
        group_by_image: If True, group items by image_id and use shared GT synsets per image
        api_url: API endpoint URL for negative mention detection (optional)
        api_key: API authorization key (optional)
    """
    from collections import defaultdict
    
    # Per question type statistics
    qtype_stats = defaultdict(lambda: {
        'num_caps': 0,
        'num_hallucinated_caps': 0,
        'hallucinated_word_count': 0,
        'vg_word_count': 0,
        'vg_words': [],
        'vg_hallucinated_words': [],
        'object_in_gts': [],
        'gt_objects': [],
        'coverage': []
    })
    
    output = {'sentences': []}
    
    # Track error samples
    error_samples = []
    
    # Group by image_id if requested
    if group_by_image:
        image_groups = defaultdict(list)
        for cap_eval in caps:
            imid = cap_eval.get('image_id', 'unknown')
            image_groups[imid].append(cap_eval)
        
        # Build shared GT synsets per image_id (merge from all items with same image_id)
        image_gt_synsets = {}
        for imid, group_items in image_groups.items():
            gt_synsets = []
            for cap_eval in group_items:
                for obj in cap_eval.get('objects', []):
                    gt_synsets.extend(obj.get('synsets', []))
                for attr in cap_eval.get('attributes', []):
                    gt_synsets.extend(attr.get('synsets', []))
                for rel in cap_eval.get('relationships', []):
                    gt_synsets.extend(rel.get("subject", {}).get('synsets', []))
                    gt_synsets.extend(rel.get("object", {}).get('synsets', []))
                for rcap in cap_eval.get('regions', []):
                    if isinstance(rcap, dict):
                        phrase = rcap.get('phrase', '')
                    else:
                        phrase = str(rcap)
                    if phrase:
                        raw_words_recap = sentence2words(phrase)
                        output_objects_recap = match_vg_list(total_synsets, raw_words_recap)
                        synset_recap = [_[1] for _ in output_objects_recap]
                        gt_synsets.extend(synset_recap)
            image_gt_synsets[imid] = list(set(gt_synsets))
        
        # Flatten groups back to list for processing
        caps = [item for group in image_groups.values() for item in group]
    
    # Track seen images for coverage when grouping
    seen_images_for_coverage = set()
    # Track matched objects per image for coverage when grouping
    image_matched_objects = defaultdict(set) if group_by_image else None
    
    for i, cap_eval in enumerate(tqdm.tqdm(caps)):
        imid = cap_eval['image_id']
        
        # Extract negated objects using API first (for all conversations)
        # This sets api_error flag if it fails
        if api_url and api_key:
            # Use full conversation context for API call
            negated_words = extract_negated_objects_from_api(cap_eval, total_synsets, api_url, api_key)
        else:
            # No API available, return empty set (no negative filtering)
            negated_words = set()
            if 'negative_mentioned_objects' not in cap_eval:
                cap_eval['negative_mentioned_objects'] = []
            cap_eval['api_error'] = False
        
        # Skip samples with API errors
        if cap_eval.get('api_error', False):
            error_samples.append(imid)
            continue
        
        # Get ground truth synsets (shared if grouping, otherwise per-item)
        if group_by_image:
            gt_synsets = image_gt_synsets.get(imid, [])
        else:
            gt_synsets = []
            for obj in cap_eval.get('objects', []):
                gt_synsets.extend(obj.get('synsets', []))
            for attr in cap_eval.get('attributes', []):
                gt_synsets.extend(attr.get('synsets', []))
            for rel in cap_eval.get('relationships', []):
                gt_synsets.extend(rel.get("subject", {}).get('synsets', []))
                gt_synsets.extend(rel.get("object", {}).get('synsets', []))
            for rcap in cap_eval.get('regions', []):
                if isinstance(rcap, dict):
                    phrase = rcap.get('phrase', '')
                else:
                    phrase = str(rcap)
                if phrase:
                    raw_words_recap = sentence2words(phrase)
                    output_objects_recap = match_vg_list(total_synsets, raw_words_recap)
                    synset_recap = [_[1] for _ in output_objects_recap]
                    gt_synsets.extend(synset_recap)
            gt_synsets = list(set(gt_synsets))
        
        # Process each conversation round separately
        conversations = cap_eval.get('conversations', [])
        if not conversations:
            # Fallback: use the joined caption if no conversations
            cap = cap_eval.get('caption', '')
            if cap:
                conversations = [{'response': cap, 'q_type': 'unknown', 'round_id': 1}]
        
        for conv_idx, conv in enumerate(conversations):
            cap = conv.get('response', '')
            q_type = conv.get('q_type', 'unknown')
            
            if not cap:
                continue
            
            # Use the negated_words already extracted (shared across all rounds for this sample)
            raw_words = sentence2words(cap)
            
            # Match words to synsets, excluding negated words
            output_objects = match_vg_list(total_synsets, raw_words, negated_words_set=negated_words)
            
            cap_dict = {
                'image_id': imid,
                'round_id': conv.get('round_id', 0),
                'q_type': q_type,
                'caption': cap,
                'vg_hallucinated_words': [],
                'vg_gt_words': gt_synsets,
                'vg_generated_words': [_[0] for _ in output_objects],
                'words': raw_words,
                'negative_mentioned_objects': cap_eval.get('negative_mentioned_objects', [])
            }
            cap_dict['metrics'] = {'CHAIRs': 0, 'CHAIRi': 0}
            
            # Count hallucinated words
            qtype_stats[q_type]['vg_word_count'] += len(raw_words)
            hallucinated = False
            vg_words_i = []
            vg_hallucinated_words_i = []
            object_in_gts = []
            
            for word, syn_class in output_objects:
                vg_words_i.append((word, syn_class))
                if is_hallucinate(syn_class, gt_synsets):
                    qtype_stats[q_type]['hallucinated_word_count'] += 1
                    cap_dict['vg_hallucinated_words'].append((word, syn_class))
                    vg_hallucinated_words_i.append((word, syn_class))
                    hallucinated = True
                else:
                    object_in_gts.append(syn_class)
            
            vg_words_i = list(set(vg_words_i))
            vg_hallucinated_words_i = list(set(vg_hallucinated_words_i))
            
            qtype_stats[q_type]['vg_words'].extend(vg_words_i)
            qtype_stats[q_type]['vg_hallucinated_words'].extend(vg_hallucinated_words_i)
            qtype_stats[q_type]['object_in_gts'].extend(list(set(object_in_gts)))
            
            # For coverage: if grouping by image, only add GT once per image per question type
            if group_by_image:
                coverage_key = (imid, q_type)
                if coverage_key not in seen_images_for_coverage:
                    qtype_stats[q_type]['gt_objects'].extend(gt_synsets)
                    seen_images_for_coverage.add(coverage_key)
                    if len(gt_synsets) > 0:
                        qtype_stats[q_type]['coverage'].append(len(list(set(object_in_gts))) / len(gt_synsets))
            else:
                qtype_stats[q_type]['gt_objects'].extend(gt_synsets)
                if len(gt_synsets) > 0:
                    qtype_stats[q_type]['coverage'].append(len(list(set(object_in_gts))) / len(gt_synsets))
            
            qtype_stats[q_type]['num_caps'] += 1
            if hallucinated:
                qtype_stats[q_type]['num_hallucinated_caps'] += 1
            
            cap_dict['metrics']['CHAIRs'] = int(hallucinated)
            cap_dict['metrics']['CHAIRi'] = 0.
            if len(raw_words) > 0:
                cap_dict['metrics']['CHAIRi'] = len(cap_dict['vg_hallucinated_words']) / float(len(raw_words))
            
            output['sentences'].append(cap_dict)
    
    # Compute overall metrics (aggregate across all question types)
    total_num_caps = sum(stats['num_caps'] for stats in qtype_stats.values())
    total_num_hallucinated_caps = sum(stats['num_hallucinated_caps'] for stats in qtype_stats.values())
    total_hallucinated_word_count = sum(stats['hallucinated_word_count'] for stats in qtype_stats.values())
    total_vg_word_count = sum(stats['vg_word_count'] for stats in qtype_stats.values())
    
    all_vg_words = []
    all_vg_hallucinated_words = []
    all_object_in_gts = []
    all_gt_objects = []
    all_coverage = []
    
    for stats in qtype_stats.values():
        all_vg_words.extend(stats['vg_words'])
        all_vg_hallucinated_words.extend(stats['vg_hallucinated_words'])
        all_object_in_gts.extend(stats['object_in_gts'])
        all_gt_objects.extend(stats['gt_objects'])
        all_coverage.extend(stats['coverage'])
    
    chair_s = (total_num_hallucinated_caps / total_num_caps * 100) if total_num_caps > 0 else 0
    chair_i = (total_hallucinated_word_count / total_vg_word_count * 100) if total_vg_word_count > 0 else 0
    chair_i_v2 = (len(all_vg_hallucinated_words) / len(all_vg_words) * 100) if len(all_vg_words) > 0 else 0
    coverage_avg = np.mean(all_coverage) * 100 if len(all_coverage) > 0 else 0
    coverage_all = (len(set(all_object_in_gts)) / len(set(all_gt_objects)) * 100) if len(set(all_gt_objects)) > 0 else 0
    
    output['overall_metrics'] = {
        'CHAIRs': chair_s,
        'CHAIRi': chair_i,
        'CHAIRi_v2': chair_i_v2,
        'Coverage_avg': coverage_avg,
        'Coverage_all': coverage_all,
        'error_samples_count': len(error_samples),
        'error_samples': error_samples
    }
    
    # Compute overall metrics per question type
    output['by_qtype'] = {}
    for q_type, stats in qtype_stats.items():
        if stats['num_caps'] == 0:
            continue
        
        chair_s = (stats['num_hallucinated_caps'] / stats['num_caps']) * 100
        chair_i = (stats['hallucinated_word_count'] / stats['vg_word_count'] * 100) if stats['vg_word_count'] > 0 else 0
        chair_i_v2 = (len(stats['vg_hallucinated_words']) / len(stats['vg_words']) * 100) if len(stats['vg_words']) > 0 else 0
        coverage_avg = np.mean(stats['coverage']) * 100 if len(stats['coverage']) > 0 else 0
        coverage_all = (len(set(stats['object_in_gts'])) / len(set(stats['gt_objects'])) * 100) if len(set(stats['gt_objects'])) > 0 else 0
        
        output['by_qtype'][q_type] = {
            'CHAIRs': chair_s,
            'CHAIRi': chair_i,
            'CHAIRi_v2': chair_i_v2,
            'Coverage_avg': coverage_avg,
            'Coverage_all': coverage_all,
            'num_caps': stats['num_caps'],
            'num_hallucinated_caps': stats['num_hallucinated_caps']
        }
    
    return output


def print_metrics_by_qtype(result):
    """Print metrics grouped by question type"""
    if 'by_qtype' not in result:
        print("No question type data available. Make sure conversations have 'q_type' field.")
        return
    
    print("\n" + "="*100)
    print("METRICS BY QUESTION TYPE")
    print("="*100)
    print("\nQuestion-Type\tCHAIRs\tCHAIRi\tCHAIRi_v2\tCoverage-avg\tCoverage-all\tSentences\tHallucinated")
    print("-"*100)
    
    for q_type in sorted(result['by_qtype'].keys()):
        metrics = result['by_qtype'][q_type]
        print(f"{q_type:15s}\t{metrics['CHAIRs']:6.1f}\t{metrics['CHAIRi']:6.1f}\t{metrics['CHAIRi_v2']:9.1f}\t"
              f"{metrics['Coverage_avg']:12.1f}\t{metrics['Coverage_all']:12.1f}\t"
              f"{metrics['num_caps']:9d}\t{metrics['num_hallucinated_caps']:12d}")
    
    print("="*100)
    
    # Report errors if any
    if 'overall_metrics' in result:
        error_count = result['overall_metrics'].get('error_samples_count', 0)
        if error_count > 0:
            print(f"\n⚠️  Warning: {error_count} sample(s) had API errors and were excluded from calculations")
            error_samples = result['overall_metrics'].get('error_samples', [])
            if len(error_samples) <= 10:
                print(f"   Error samples: {error_samples}")
            else:
                print(f"   Error samples (first 10): {error_samples[:10]}...")
                print(f"   (and {len(error_samples) - 10} more)")
    
    print()






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
    parser.add_argument("--by_qtype", action='store_true', 
                       help='Report metrics grouped by question type')
    parser.add_argument("--group-by-image", action='store_true',
                       help='Group items by image_id and use shared ground truth synsets per image')
    parser.add_argument("--api-url", type=str, default=None,
                       help='API endpoint URL for negative mention detection (e.g., https://.../v1/chat/completions)')
    parser.add_argument("--api-key", type=str, default=None,
                       help='API authorization key (Bearer token)')
    parser.add_argument("--model", type=str, default="Qwen3-30B-A3B-Instruct-2507",
                       help='Model name to use for API calls')
    args = parser.parse_args()
    
    with open(args.object_synsets) as object_dict_file:
        object_dict = json.load(object_dict_file)
        object_dict = {key.split('.')[0]: value for key, value in object_dict.items()}

    object_dict = {key: value for key, value in object_dict.items() if is_physical_object(value)}
    
    stemmed_object_dict = {}
    
    for word, synset in object_dict.items():
        stemmed_object_dict[ps.stem(word)] = (word, synset)
    
    data = json.load(open(args.cap_file, 'r'))
    
    # Handle metadata first
    for sample in data:
        for k, v in sample.get("metadata", {}).items():
            sample[k] = v
    
    if args.by_qtype:
        # Process conversations separately for question type analysis
        chair_result = compute_chair_by_qtype(data, stemmed_object_dict, group_by_image=args.group_by_image, 
                                             api_url=args.api_url, api_key=args.api_key)
        print_metrics(chair_result)
        print_metrics_by_qtype(chair_result)
        save_hallucinated_words(args.cap_file, chair_result)
    else:
        # Concatenate responses based on grouping flag
        if args.group_by_image:
            # Group by image_id and concatenate all responses from all conversations
            from collections import defaultdict
            image_groups = defaultdict(list)
            for sample in data:
                imid = sample.get('image_id', 'unknown')
                image_groups[imid].append(sample)
            
            # Create new data list with concatenated captions per image
            processed_data = []
            for imid, group_items in image_groups.items():
                # Collect all responses from all conversations in all items with same image_id
                all_responses = []
                # Use first item as base, merge GT from all items
                base_item = group_items[0].copy()
                for item in group_items:
                    for conv in item.get('conversations', []):
                        all_responses.append(conv.get('response', ''))
                
                base_item['caption'] = " ".join(all_responses)
                # Merge objects, attributes, relationships, regions from all items
                all_objects = []
                all_attributes = []
                all_relationships = []
                all_regions = []
                for item in group_items:
                    all_objects.extend(item.get('objects', []))
                    all_attributes.extend(item.get('attributes', []))
                    all_relationships.extend(item.get('relationships', []))
                    all_regions.extend(item.get('regions', []))
                
                base_item['objects'] = all_objects
                base_item['attributes'] = all_attributes
                base_item['relationships'] = all_relationships
                base_item['regions'] = all_regions
                processed_data.append(base_item)
            
            data = processed_data
            # Don't group again in compute_chair since we've already grouped
            chair_result = compute_chair(data, stemmed_object_dict, group_by_image=False, 
                                       api_url=args.api_url, api_key=args.api_key, model=args.model)
        else:
            # Original behavior: join all responses in all rounds for each item
            for sample in data:
                responses = [conv["response"] for conv in sample.get("conversations", [])]
                sample["caption"] = " ".join(responses)
            
            chair_result = compute_chair(data, stemmed_object_dict, group_by_image=False,
                                       api_url=args.api_url, api_key=args.api_key, model=args.model)
        print_metrics(chair_result)
        save_hallucinated_words(args.cap_file, chair_result)
