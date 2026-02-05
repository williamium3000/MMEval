import json
import tqdm
import os
import sys
import time
from PIL import Image
import argparse
from openai import OpenAI

API_RETRY_WAIT_SEC = 5
API_MAX_RETRIES = 5
API_TIMEOUT_SEC = 300  # 5 minutes, wait for response

# Ensure project root is on path so utils/infer can be imported
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Load .env from project root for remote API
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(_project_root, ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

from utils.query_utils import generate_dsg
from utils.parse_utils import parse_tuple_output, parse_question_output
from functools import partial

# Remote API endpoint — read from .env
REMOTE_API_URL = os.getenv("REMOTE_API_URL", "")
if REMOTE_API_URL:
    # Normalize: remove /chat/completions if present, ensure ends with /v1
    REMOTE_BASE_URL = REMOTE_API_URL.rstrip("/chat/completions").rstrip("/v1")
    if not REMOTE_BASE_URL.endswith("/v1"):
        REMOTE_BASE_URL = REMOTE_BASE_URL + "/v1"
else:
    REMOTE_BASE_URL = ""
REMOTE_API_KEY = os.getenv("REMOTE_API_KEY", "")
REMOTE_MODEL = os.getenv("REMOTE_API_MODEL", "Qwen3-30B-A3B-Instruct-2507")

# Local vLLM server endpoint (matches scripts/host_qwen3_30b_gpu7.sh port 8003)
LOCAL_BASE_URL = "http://localhost:8003/v1"
LOCAL_API_KEY = ""  # Dummy key for vLLM (not used)
LOCAL_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # Full model path as hosted by vLLM

# Initialize client only if remote API URL is configured
if REMOTE_BASE_URL:
    client = OpenAI(api_key=REMOTE_API_KEY, base_url=REMOTE_BASE_URL, timeout=API_TIMEOUT_SEC)
else:
    client = None


def get_llm_response(
        prompt,
        model=None,
        temperature=0,
        return_response=False,
        max_tokens=500,
):
    if model is None:
        model = REMOTE_MODEL

    last_err = None
    for attempt in range(API_MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=API_TIMEOUT_SEC,
            )
            if return_response:
                return completion
            return completion.choices[0].message.content
        except Exception as e:
            last_err = e
            if attempt < API_MAX_RETRIES - 1:
                print(f"API error (attempt {attempt + 1}/{API_MAX_RETRIES}), retrying in {API_RETRY_WAIT_SEC}s: {e}")
                time.sleep(API_RETRY_WAIT_SEC)
            else:
                print(f"API error after {API_MAX_RETRIES} attempts: {e}")
                raise last_err


def get_llm_response_batch(prompts, model=None, max_workers=None):
    """Send multiple prompts to LLM sequentially (one after another).
    
    Args:
        prompts: List of prompts or dict of {id: prompt}
        model: Model name
        max_workers: Ignored (kept for API compatibility).
        
    Returns:
        List of responses (if input is list) or dict of {id: response} (if input is dict)
    """
    if model is None:
        model = REMOTE_MODEL

    is_dict = isinstance(prompts, dict)
    if is_dict:
        prompt_items = list(prompts.items())
    else:
        prompt_items = [(i, p) for i, p in enumerate(prompts)]
    
    results = {}
    for idx, prompt in prompt_items:
        try:
            results[idx] = get_llm_response(prompt, model=model)
        except Exception as e:
            print(f"Error calling LLM for {idx}: {e}")
            results[idx] = ""
    
    if is_dict:
        return results
    else:
        return [results[i] for i in range(len(prompts))]


BATCH_ANSWER_EXTRACTION_PROMPT = """Given the following transcript of a conversation and a list of yes/no questions, extract whether the transcript contains information that can answer each question.

Transcript:
{transcript}

Questions:
{questions}

For each question, determine:
1. Does the transcript contain relevant information to answer this question?
2. If yes, what is the answer (Yes/No/Unknown)?

Respond in JSON format as a list, with one object per question in the same order:
[
  {{"id": <question_id>, "has_answer": true/false, "answer": "Yes"/"No"/"Unknown", "evidence": "relevant text from transcript"}},
  ...
]

Important:
- Answer "Yes" if the transcript explicitly confirms the statement in the question
- Answer "No" if the transcript explicitly contradicts or denies the statement
- Answer "Unknown" if the transcript doesn't contain enough information to determine
- If has_answer is false, set answer to "Unknown"
- Make sure to return answers for ALL questions in the same order
"""


def extract_answers_batch_from_transcript(questions_dict, transcript, get_response):
    """Extract answers to multiple questions from the conversation transcript using a single LLM call.
    
    Args:
        questions_dict: Dict mapping question_id -> question text
        transcript: The conversation transcript text
        get_response: Function to call LLM
        
    Returns:
        Dict mapping question_id -> {has_answer, answer, evidence}
    """
    import re
    
    # Format questions as numbered list
    questions_list = []
    id_order = []
    for qid, question in questions_dict.items():
        questions_list.append(f"{qid}. {question}")
        id_order.append(qid)
    
    questions_str = "\n".join(questions_list)
    
    prompt = BATCH_ANSWER_EXTRACTION_PROMPT.format(
        transcript=transcript,
        questions=questions_str
    )
    
    response = get_response(prompt)
    
    # Parse JSON response
    results = {}
    try:
        # Try to extract JSON array from the response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            parsed = json.loads(json_match.group())
            for item in parsed:
                qid = item.get("id")
                if qid is not None:
                    # Handle both int and string ids
                    if isinstance(qid, str) and qid.isdigit():
                        qid = int(qid)
                    results[qid] = {
                        "has_answer": item.get("has_answer", False),
                        "answer": item.get("answer", "Unknown"),
                        "evidence": item.get("evidence", "")
                    }
    except (json.JSONDecodeError, AttributeError) as e:
        pass
    
    # Fill in missing questions with default values
    for qid in questions_dict.keys():
        if qid not in results:
            results[qid] = {"has_answer": False, "answer": "Unknown", "evidence": ""}
    
    return results


def format_conversation_as_transcript(conversation):
    """Format conversation turns into a readable transcript string.
    
    Supports multiple formats:
    1. Standard format: {"role": "user/assistant", "content": "..."}
    2. Response format: {"role": "user/assistant", "response": "..."}
    3. Prompt-response format: {"prompt": "...", "response": "..."}
    
    Args:
        conversation: List of conversation turns
        
    Returns:
        Formatted transcript string
    """
    transcript_parts = []
    for i, turn in enumerate(conversation):
        # Determine role
        role = turn.get("role", "")
        if not role:
            # Infer role from structure
            if "prompt" in turn and "response" in turn:
                role = "qa_pair"
            else:
                role = "unknown"
        
        # Format based on structure
        if role == "qa_pair" or ("prompt" in turn and "response" in turn):
            # Prompt-response format (e.g., output/context/single/*.json)
            prompt = turn.get("prompt", "")
            response = turn.get("response", "")
            transcript_parts.append(f"[Turn {i+1}] QUESTION: {prompt}")
            transcript_parts.append(f"[Turn {i+1}] ANSWER: {response}")
        else:
            # Standard format
            content = turn.get("content") or turn.get("response", "")
            transcript_parts.append(f"[Turn {i+1}] {role.upper()}: {content}")
    
    return "\n\n".join(transcript_parts)


GROUND_TRUTH_VERIFICATION_PROMPT = """Given the image annotation (ground truth) and a list of yes/no questions about the image, determine the correct answer for each question based ONLY on the annotation.

Image Annotation:
{annotation}

Questions:
{questions}

For each question, determine the ground truth answer based on the annotation:
- Answer "Yes" if the annotation confirms the statement in the question
- Answer "No" if the annotation contradicts or does not support the statement
- Answer "Unknown" if the annotation doesn't contain enough information to determine

CRITICAL RULES:
1. If the object(s) mentioned in the question do NOT exist in the image annotation (not listed in objects), you MUST answer "No". Do not use "Unknown" for absent objects—treat them as incorrect.
2. If the objects exist but a specific relationship between them is not mentioned in the annotation, you should answer "Unknown" (not "No").
3. If the objects exist but a specific attribute is not mentioned, you should answer "Unknown" (not "No").

Respond in JSON format as a list:
[
  {{"id": "<question_id>", "gt_answer": "Yes"/"No"/"Unknown", "reasoning": "brief explanation"}},
  ...
]

IMPORTANT: The "id" field must be the exact question ID string (e.g., "0_1", "1_2"), not just a number.
"""


def format_vg_annotation(sample):
    """Format Visual Genome style annotation (objects, attributes, relationships) as readable string.
    
    Supports multiple formats:
    1. Flat format: sample["objects"], sample["attributes"], sample["relationships"]
    2. Nested format: sample["sg"]["objects"], sample["sg"]["relationships"]
    3. With metadata: sample["metadata"]["objects"], etc.
    
    Args:
        sample: Sample dict containing annotation data
        
    Returns:
        Formatted annotation string
    """
    annotation_parts = []
    
    # Handle different nesting formats
    if "metadata" in sample:
        sample = sample["metadata"]
    
    # Check for "sg" nested format (e.g., output/context/single/*.json)
    if "sg" in sample:
        sg_data = sample["sg"]
    else:
        sg_data = sample
    
    # Format objects - handle both list and dict formats
    objects_data = sg_data.get("objects", sample.get("objects"))
    if objects_data:
        object_names = []
        attr_descriptions = []
        
        # Handle dict format (keyed by object_id)
        if isinstance(objects_data, dict):
            for obj_id, obj in objects_data.items():
                names = obj.get("names", [])
                if names:
                    obj_name = names[0]
                    object_names.append(obj_name)
                    # Also extract attributes if present
                    attrs = obj.get("attributes")
                    if attrs:
                        if isinstance(attrs, list):
                            attr_descriptions.append(f"{obj_name}: {', '.join(attrs)}")
                        else:
                            attr_descriptions.append(f"{obj_name}: {attrs}")
        # Handle list format
        elif isinstance(objects_data, list):
            for obj in objects_data:
                names = obj.get("names", [])
                if names:
                    obj_name = names[0]
                    object_names.append(obj_name)
                    # Also extract attributes if present
                    attrs = obj.get("attributes")
                    if attrs:
                        if isinstance(attrs, list):
                            attr_descriptions.append(f"{obj_name}: {', '.join(attrs)}")
                        else:
                            attr_descriptions.append(f"{obj_name}: {attrs}")
        
        if object_names:
            annotation_parts.append(f"Objects in image: {', '.join(set(object_names))}")
        if attr_descriptions:
            annotation_parts.append(f"Object attributes:\n  " + "\n  ".join(attr_descriptions))
    
    # Format separate attributes field (if exists and not already processed)
    attributes_data = sg_data.get("attributes", sample.get("attributes"))
    if attributes_data and isinstance(attributes_data, list):
        attr_descriptions = []
        for attr_obj in attributes_data:
            names = attr_obj.get("names", [])
            attrs = attr_obj.get("attributes")
            if names and attrs:
                obj_name = names[0]
                if isinstance(attrs, list):
                    attr_descriptions.append(f"{obj_name}: {', '.join(attrs)}")
                elif attrs:
                    attr_descriptions.append(f"{obj_name}: {attrs}")
        if attr_descriptions:
            annotation_parts.append(f"Object attributes:\n  " + "\n  ".join(attr_descriptions))
    
    # Format relationships
    relationships_data = sg_data.get("relationships", sample.get("relationships"))
    if relationships_data:
        rel_descriptions = []
        for rel in relationships_data:
            predicate = rel.get("predicate", "")
            subject = rel.get("subject", {})
            obj = rel.get("object", {})
            
            # Handle both nested names and direct object_id references
            if isinstance(subject, dict):
                subj_names = subject.get("names", [])
            else:
                subj_names = []
            
            if isinstance(obj, dict):
                obj_names = obj.get("names", [])
            else:
                obj_names = []
            
            if subj_names and obj_names and predicate:
                rel_descriptions.append(f"{subj_names[0]} {predicate} {obj_names[0]}")
        if rel_descriptions:
            annotation_parts.append(f"Relationships:\n  " + "\n  ".join(rel_descriptions))
    
    return "\n\n".join(annotation_parts) if annotation_parts else ""


def verify_answers_with_annotation(questions_dict, annotation, get_response):
    """Verify answers against ground truth annotation using LLM.
    
    Args:
        questions_dict: Dict mapping question_id -> question text
        annotation: The ground truth annotation - can be:
            - Sample dict with 'objects', 'attributes', 'relationships' (VG format)
            - Dict with 'instances', 'captions' (COCO format)
            - Any other dict or string
        get_response: Function to call LLM
        
    Returns:
        Dict mapping question_id -> {gt_answer, reasoning}
    """
    import re
    
    # Format annotation as readable string based on format
    if isinstance(annotation, dict):
        # Check for VG format - including "sg" nested format
        if any(key in annotation for key in ["objects", "attributes", "relationships", "sg"]):
            annotation_str = format_vg_annotation(annotation)
        # Check for COCO format (instances, captions)
        elif "instances" in annotation or "captions" in annotation:
            annotation_parts = []
            if "instances" in annotation:
                objects = [inst.get("category", "") for inst in annotation["instances"]]
                annotation_parts.append(f"Objects in image: {', '.join(objects)}")
            if "captions" in annotation:
                annotation_parts.append(f"Image captions: {'; '.join(annotation['captions'])}")
            annotation_str = "\n".join(annotation_parts)
        else:
            annotation_str = str(annotation)
        
        if not annotation_str:
            annotation_str = str(annotation)
    else:
        annotation_str = str(annotation)
    
    # Format questions
    questions_list = []
    for qid, question in questions_dict.items():
        questions_list.append(f"{qid}. {question}")
    questions_str = "\n".join(questions_list)
    
    prompt = GROUND_TRUTH_VERIFICATION_PROMPT.format(
        annotation=annotation_str,
        questions=questions_str
    )
    
    response = get_response(prompt)
    
    # Parse JSON response
    results = {}
    try:
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            parsed = json.loads(json_match.group())
            for item in parsed:
                qid = item.get("id")
                if qid is not None:
                    # Keep string IDs as-is (e.g., "0_1")
                    results[qid] = {
                        "gt_answer": item.get("gt_answer", "Unknown"),
                        "reasoning": item.get("reasoning", "")
                    }
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Fill in missing questions
    for qid in questions_dict.keys():
        if qid not in results:
            results[qid] = {"gt_answer": "Unknown", "reasoning": ""}
    
    return results


def normalize_answer(answer):
    """Normalize answer to yes/no/unknown."""
    if not answer:
        return "unknown"
    
    answer_lower = answer.lower().strip()
    
    # Check for explicit unknown indicators
    unknown_indicators = ["unknown", "unclear", "cannot determine", "not enough information", 
                          "n/a", "not available", "cannot be determined", "indeterminate"]
    for indicator in unknown_indicators:
        if indicator in answer_lower:
            return "unknown"
    
    # Check for yes/no
    if "yes" in answer_lower:
        return "yes"
    elif "no" in answer_lower:
        return "no"
    else:
        return "unknown"


def compute_accuracy(extracted_answer, gt_answer):
    """Compute whether extracted answer matches ground truth.
    
    Args:
        extracted_answer: Answer extracted from transcript (Yes/No/Unknown)
        gt_answer: Ground truth answer (Yes/No/Unknown)
        
    Returns:
        dict with is_correct, extracted_norm, gt_norm, skip_reason
        - is_correct: True/False/None
        - skip_reason: Additional info about the comparison (if applicable)
    
    Note: 
        - gt_unknown: counted as incorrect (is_correct=False)
        - extracted_unknown: skipped (is_correct=None)
    """
    extracted_norm = normalize_answer(extracted_answer)
    gt_norm = normalize_answer(gt_answer)
    
    skip_reason = None
    is_correct = None
    
    if gt_norm == "unknown":
        # GT is unknown - count as incorrect (model made a claim that can't be verified)
        skip_reason = "gt_unknown"
        is_correct = False
    elif extracted_norm == "unknown":
        # Extracted answer is unknown but gt is known - skip (not counted)
        skip_reason = "extracted_unknown"
        is_correct = None  # Skip this question
        is_correct = None  # Skip by default, but track it
    else:
        # Both are definite (yes or no)
        is_correct = (extracted_norm == gt_norm)
    
    return {
        "is_correct": is_correct,
        "extracted_norm": extracted_norm,
        "gt_norm": gt_norm,
        "skip_reason": skip_reason
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv_script', type=str, default="output/context/single/llava-1.5-7b-hf_single.json")
    parser.add_argument('--outfile', type=str, default="output/context/pope/llava-1.5-7b-hf_single_pope.json")
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--pope_model_name', type=str, default=REMOTE_MODEL,
                        help="LLM model name (default: Qwen3-30B-A3B-Instruct-2507)")
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--sample_num', type=int, default=81)
    parser.add_argument('--start_idx', type=int, default=0,
                        help="Start index for processing samples")
    parser.add_argument('--extract_from_transcript', action="store_true",
                        help="Extract answers from transcript instead of calling VLM model")
    parser.add_argument('--response_key', type=str, default="response",
                        help="Key for VLM response in conversation turns (e.g., 'response' or 'content')")
    parser.add_argument('--verify_with_annotation', action="store_true",
                        help="Verify extracted answers against ground truth annotation")
    parser.add_argument('--verify_only', action="store_true",
                        help="Only run verification. Skip DSG generation and extraction. Input must have dsg_qa. Implies --verify_with_annotation. "
                             "Only samples in [start_idx, start_idx+sample_num) get judge fields (gt_answer, is_correct, etc.); use --start_idx 0 for the first sample.")
    parser.add_argument('--annotation_key', type=str, default="image_info",
                        help="Key for annotation in sample (e.g., 'image_info', 'annotation', 'gt')")
    parser.add_argument('--batch_size', type=int, default=5,
                        help="Number of samples per batch")
    parser.add_argument('--local', action="store_true",
                        help="Use local vLLM server on localhost:8003 instead of remote API")

    args = parser.parse_args()

    if args.verify_only:
        args.verify_with_annotation = True

    # Reinitialize client if using local server
    if args.local:
        client = OpenAI(api_key=LOCAL_API_KEY, base_url=LOCAL_BASE_URL, timeout=API_TIMEOUT_SEC)
        # Override pope_model_name to use the full model path for local vLLM
        if not args.pope_model_name or args.pope_model_name == REMOTE_MODEL:
            args.pope_model_name = LOCAL_MODEL
        print(f"Using local vLLM server at {LOCAL_BASE_URL} with model {args.pope_model_name}")

    # Prevent overwriting input: refuse if conv_script and outfile are the same
    conv_path = os.path.abspath(args.conv_script)
    out_path = os.path.abspath(args.outfile)
    if conv_path == out_path:
        raise SystemExit(
            "Error: --conv_script and --outfile must point to different files. "
            "Using the same path would overwrite the input. Use a different --outfile."
        )

    samples = json.load(open(args.conv_script, "r"))

    get_response = partial(get_llm_response, model=args.pope_model_name)
    get_response_batch = partial(get_llm_response_batch, model=args.pope_model_name)

    # Only load VLM model if not extracting from transcript and not verify_only
    eval_model_fn = None
    if not args.extract_from_transcript and not args.verify_only:
        from infer.infer_llava import load_model, eval_model
        eval_model_fn = eval_model
        model_name, tokenizer, model, image_processor, context_len = load_model(args.model_path, args.model_base)
        model_path = args.model_path

    # Select samples to process
    samples_to_process = samples[args.start_idx:args.start_idx + args.sample_num]
    
    print(f"Processing {len(samples_to_process)} samples (from index {args.start_idx})")
    print(f"Batch size: {args.batch_size}")
    if args.verify_only:
        print("Verify-only mode: skipping DSG generation and extraction, running verification only.")

    # Process samples in batches
    for batch_start in tqdm.tqdm(range(0, len(samples_to_process), args.batch_size), desc="Batches"):
        batch_samples = samples_to_process[batch_start:batch_start + args.batch_size]
        
        # ============================================================
        # Phase 1: Prepare all prompts for DSG generation (batch)
        # ============================================================
        all_dsg_prompts = {}  # {(sample_idx, turn_idx, task): prompt}
        sample_data = {}  # Store intermediate data for each sample
        
        for sample_idx, sample in enumerate(batch_samples):
            conversation = sample["conversations"]
            sample_data[sample_idx] = {
                "conversation": conversation,
                "transcript": format_conversation_as_transcript(conversation) if args.extract_from_transcript else "",
                "id2prompts": {}
            }
            
            for turn_idx, turn in enumerate(conversation):
                vlm_response = turn.get(args.response_key) or turn.get("content", "")
                sample_data[sample_idx]["id2prompts"][f'custom_{turn_idx}'] = {'input': vlm_response}
        
        if not args.verify_only:
            # Generate DSG for all samples in batch (sequential)
            def process_dsg_for_sample(sample_idx):
                id2prompts = sample_data[sample_idx]["id2prompts"]
                try:
                    id2tuple_outputs, id2question_outputs = generate_dsg(
                        id2prompts,
                        generate_fn=get_response,
                        verbose=False
                    )
                    return sample_idx, id2tuple_outputs, id2question_outputs
                except Exception as e:
                    print(f"Error in DSG generation for sample {sample_idx}: {e}")
                    return sample_idx, {}, {}
            
            dsg_results = [process_dsg_for_sample(i) for i in range(len(batch_samples))]
            
            for sample_idx, id2tuple_outputs, id2question_outputs in dsg_results:
                sample_data[sample_idx]["id2tuple_outputs"] = id2tuple_outputs
                sample_data[sample_idx]["id2question_outputs"] = id2question_outputs
        
        # ============================================================
        # Phase 2: Collect all questions and prepare batch prompts
        # ============================================================
        all_extraction_prompts = {}  # {sample_idx: prompt}
        all_verification_prompts = {}  # {sample_idx: prompt}
        
        for sample_idx, sample in enumerate(batch_samples):
            conversation = sample_data[sample_idx]["conversation"]
            
            # Collect all questions: from dsg_qa (verify_only) or from DSG outputs
            all_questions = {}
            all_tuples = {}
            
            if args.verify_only:
                for turn_idx, turn in enumerate(conversation):
                    for qa in turn.get("dsg_qa", []):
                        qid = qa.get("index")
                        composite_key = f"{turn_idx}_{qid}"
                        all_questions[composite_key] = qa.get("question", "")
                        all_tuples[composite_key] = qa.get("tuple", "")
            else:
                id2tuple_outputs = sample_data[sample_idx].get("id2tuple_outputs", {})
                id2question_outputs = sample_data[sample_idx].get("id2question_outputs", {})
                for turn_idx, turn in enumerate(conversation):
                    try:
                        qid2tuple = parse_tuple_output(id2tuple_outputs.get(f'custom_{turn_idx}', {}).get('output', ''))
                        qid2question = parse_question_output(id2question_outputs.get(f'custom_{turn_idx}', {}).get('output', ''))
                    except Exception:
                        qid2tuple = {}
                        qid2question = {}
                    for qid, question in qid2question.items():
                        composite_key = f"{turn_idx}_{qid}"
                        all_questions[composite_key] = question
                        all_tuples[composite_key] = qid2tuple.get(qid, "")
            
            sample_data[sample_idx]["all_questions"] = all_questions
            sample_data[sample_idx]["all_tuples"] = all_tuples
            
            # Prepare extraction prompt (skip when verify_only)
            if args.extract_from_transcript and not args.verify_only and all_questions:
                questions_list = [f"{qid}. {q}" for qid, q in all_questions.items()]
                all_extraction_prompts[sample_idx] = BATCH_ANSWER_EXTRACTION_PROMPT.format(
                    transcript=sample_data[sample_idx]["transcript"],
                    questions="\n".join(questions_list)
                )
            
            # Prepare verification prompt
            if args.verify_with_annotation and all_questions:
                if args.annotation_key and args.annotation_key in sample:
                    annotation = sample.get(args.annotation_key, {})
                elif "sg" in sample:
                    annotation = sample
                elif any(key in sample for key in ["objects", "attributes", "relationships"]):
                    annotation = sample
                else:
                    annotation = sample.get(args.annotation_key, {})
                
                if annotation:
                    annotation_str = format_vg_annotation(annotation) if isinstance(annotation, dict) else str(annotation)
                    questions_list = [f"{qid}. {q}" for qid, q in all_questions.items()]
                    all_verification_prompts[sample_idx] = GROUND_TRUTH_VERIFICATION_PROMPT.format(
                        annotation=annotation_str,
                        questions="\n".join(questions_list)
                    )
                    sample_data[sample_idx]["annotation"] = annotation
        
        # ============================================================
        # Phase 3: Batch LLM calls for extraction and verification
        # ============================================================
        extraction_responses = {}
        verification_responses = {}
        
        if all_extraction_prompts:
            if args.verbose:
                print(f"  Sending {len(all_extraction_prompts)} extraction prompts...")
            extraction_responses = get_response_batch(all_extraction_prompts)
        
        if all_verification_prompts:
            if args.verbose:
                print(f"  Sending {len(all_verification_prompts)} verification prompts...")
            verification_responses = get_response_batch(all_verification_prompts)
        
        # ============================================================
        # Phase 4: Parse responses and compute accuracy for each sample
        # ============================================================
        for sample_idx, sample in enumerate(batch_samples):
            conversation = sample_data[sample_idx]["conversation"]
            all_questions = sample_data[sample_idx].get("all_questions", {})
            all_tuples = sample_data[sample_idx].get("all_tuples", {})
            id2tuple_outputs = sample_data[sample_idx].get("id2tuple_outputs", {})
            id2question_outputs = sample_data[sample_idx].get("id2question_outputs", {})
            
            # Parse extraction results
            batch_results = {}
            if sample_idx in extraction_responses:
                import re
                response = extraction_responses[sample_idx]
                try:
                    json_match = re.search(r'\[[\s\S]*\]', response)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        for item in parsed:
                            qid = item.get("id")
                            if qid is not None:
                                batch_results[qid] = {
                                    "has_answer": item.get("has_answer", False),
                                    "answer": item.get("answer", "Unknown"),
                                    "evidence": item.get("evidence", "")
                                }
                except:
                    pass
            
            # Parse verification results
            gt_results = {}
            if sample_idx in verification_responses:
                import re
                response = verification_responses[sample_idx]
                try:
                    json_match = re.search(r'\[[\s\S]*\]', response)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        for item in parsed:
                            qid = item.get("id")
                            if qid is not None:
                                gt_results[qid] = {
                                    "gt_answer": item.get("gt_answer", "Unknown"),
                                    "reasoning": item.get("reasoning", "")
                                }
                except:
                    pass
            
            # Step 3: Distribute results back to turns and compute accuracy
            # Image-level stats
            image_correct = 0
            image_total = 0
            image_gt_unknown = 0
            image_extracted_unknown = 0
            image_all_questions = 0
            
            # Store all question-level accuracy judgments for this sample
            all_question_judgments = []
            
            for index, turn in enumerate(conversation):
                if args.verify_only:
                    qid2question = {qa["index"]: qa["question"] for qa in turn.get("dsg_qa", [])}
                    qid2qa = {qa["index"]: qa for qa in turn.get("dsg_qa", [])}
                else:
                    try:
                        qid2question = parse_question_output(
                            sample_data[sample_idx].get("id2question_outputs", {}).get(f'custom_{index}', {}).get('output', '')
                        )
                    except Exception:
                        qid2question = {}
                    qid2qa = {}
                
                # Skip turn if no questions for this turn
                if not qid2question:
                    if args.verbose:
                        print(f"    Skipping turn {index}: no questions for this turn")
                    turn["dsg_no_questions"] = True
                    continue
                
                # Response-level stats
                response_correct = 0
                response_total = 0
                response_gt_unknown = 0
                response_extracted_unknown = 0
                response_all_questions = 0
                
                pope_qa = []
                for qid, question in qid2question.items():
                    composite_key = f"{index}_{qid}"
                    
                    if args.verify_only:
                        qa = qid2qa.get(qid, {})
                        extracted_answer = qa.get("answer", "Unknown")
                        qa_item = {
                            "index": qid,
                            "question": question,
                            "answer": extracted_answer,
                            "has_answer": qa.get("has_answer", False),
                            "evidence": qa.get("evidence", ""),
                            "tuple": all_tuples.get(composite_key, qa.get("tuple", ""))
                        }
                    elif args.extract_from_transcript:
                        extraction_result = batch_results.get(composite_key, {})
                        extracted_answer = extraction_result.get("answer", "Unknown")
                        qa_item = {
                            "index": qid,
                            "question": question,
                            "answer": extracted_answer,
                            "has_answer": extraction_result.get("has_answer", False),
                            "evidence": extraction_result.get("evidence", ""),
                            "tuple": all_tuples.get(composite_key, "")
                        }
                    else:
                        # Call VLM model (eval_model_fn only set when not extract_from_transcript/verify_only)
                        image_file = Image.open(sample["image"]).convert("RGB")
                        extracted_answer = eval_model_fn(model_name, tokenizer, model, image_processor, context_len, type('Args', (), {
                            "model_path": model_path,
                            "model_base": None,
                            "model_name": model_name,
                            "query": question,
                            "conv_mode": None,
                            "image_file": image_file,
                            "sep": ",",
                            "load_in_8bit": False,
                            "load_in_4bit": False,
                            "temperature": 0.0,
                            "top_p": None,
                            "num_beams": 1,
                            "max_new_tokens": 512
                        })())
                        qa_item = {
                            "index": qid,
                            "question": question,
                            "answer": extracted_answer,
                            "tuple": all_tuples.get(composite_key, "")
                        }
                    
                    # Add ground truth verification results (always when verify_with_annotation)
                    if args.verify_with_annotation:
                        gt_info = gt_results.get(composite_key, {"gt_answer": "Unknown", "reasoning": "no gt from verification"})
                        gt_answer = gt_info.get("gt_answer", "Unknown")
                        
                        accuracy_info = compute_accuracy(extracted_answer, gt_answer)
                        
                        qa_item["gt_answer"] = gt_answer
                        qa_item["gt_reasoning"] = gt_info.get("reasoning", "")
                        qa_item["is_correct"] = accuracy_info["is_correct"]
                        qa_item["skip_reason"] = accuracy_info.get("skip_reason")
                        qa_item["extracted_norm"] = accuracy_info["extracted_norm"]
                        qa_item["gt_norm"] = accuracy_info["gt_norm"]
                        
                        # Count all questions
                        response_all_questions += 1
                        image_all_questions += 1
                        
                        # Store question-level judgment
                        all_question_judgments.append({
                            "turn_index": index,
                            "question_id": qid,
                            "question": question,
                            "tuple": all_tuples.get(composite_key, ""),
                            "extracted_answer": extracted_answer,
                            "extracted_norm": accuracy_info["extracted_norm"],
                            "gt_answer": gt_answer,
                            "gt_norm": accuracy_info["gt_norm"],
                            "is_correct": accuracy_info["is_correct"],
                            "skip_reason": accuracy_info.get("skip_reason"),
                            "evidence": qa_item.get("evidence", ""),
                            "gt_reasoning": gt_info.get("reasoning", "")
                        })
                        
                        # Track response-level accuracy stats
                        skip_reason = accuracy_info.get("skip_reason")
                        if skip_reason == "gt_unknown":
                            response_gt_unknown += 1
                            image_gt_unknown += 1
                        elif skip_reason == "extracted_unknown":
                            response_extracted_unknown += 1
                            image_extracted_unknown += 1
                        else:
                            response_total += 1
                            image_total += 1
                            if accuracy_info["is_correct"]:
                                response_correct += 1
                                image_correct += 1
                    
                    pope_qa.append(qa_item)
                
                turn["dsg_qa"] = pope_qa
                
                # Store response-level accuracy and stats
                if args.verify_with_annotation:
                    turn["dsg_response_total"] = response_total
                    turn["dsg_response_gt_unknown"] = response_gt_unknown
                    turn["dsg_response_extracted_unknown"] = response_extracted_unknown
                    turn["dsg_response_correct"] = response_correct
                    if response_total > 0:
                        turn["dsg_response_accuracy"] = response_correct / response_total
                
                if args.verbose:
                    print(f"\n--- Turn {index} DSG Questions and Answers ---")
                    for qa in pope_qa:
                        print(f"  Q: {qa['question']}")
                        print(f"  A (extracted): {qa['answer']} -> {qa.get('extracted_norm', 'N/A')}")
                        if args.verify_with_annotation:
                            print(f"  A (ground truth): {qa.get('gt_answer', 'N/A')} -> {qa.get('gt_norm', 'N/A')}")
                            if qa.get('skip_reason'):
                                print(f"  Skipped: {qa.get('skip_reason')}")
                            else:
                                print(f"  Correct: {qa.get('is_correct', 'N/A')}")
                        if qa.get('evidence'):
                            print(f"  Evidence: {qa['evidence']}")
                        print()
                    
                    if args.verify_with_annotation:
                        print(f"  [Response-level] Total: {response_all_questions}, GT Unknown (skip): {response_gt_unknown}, Extracted Unknown (skip): {response_extracted_unknown}, Evaluated: {response_total}")
                        if response_total > 0:
                            print(f"  [Response-level] Accuracy: {response_correct}/{response_total} = {response_correct/response_total:.2%}")
        
            # Store image-level accuracy and all question judgments
            if args.verify_with_annotation:
                # Store all question-level judgments
                sample["dsg_question_judgments"] = all_question_judgments
                
                # Store detailed stats
                sample["dsg_image_total"] = image_total
                sample["dsg_image_gt_unknown"] = image_gt_unknown
                sample["dsg_image_extracted_unknown"] = image_extracted_unknown
                sample["dsg_image_correct"] = image_correct
                
                if image_total > 0:
                    sample["dsg_image_accuracy"] = image_correct / image_total
                    
                if args.verbose:
                    print(f"\n[Image-level] Total: {image_total}, GT Unknown (skip): {image_gt_unknown}, Extracted Unknown (skip): {image_extracted_unknown}")
                    if image_total > 0:
                        print(f"[Image-level] Accuracy: {image_correct}/{image_total} = {sample['dsg_image_accuracy']:.2%}")
        
        # Save intermediate results after each batch
        with open(args.outfile, "w") as f:
            json.dump(samples, f, indent=4)

    print(f"\nResults saved to {args.outfile}")
    
    # Print overall statistics
    if args.verify_with_annotation:
        processed_samples = samples_to_process
        
        # Track samples/turns with no questions
        samples_with_no_questions = 0
        turns_with_no_questions = 0
        
        # Image-level statistics - recalculate from question judgments
        total_generated = 0
        total_evaluated = 0
        total_gt_unknown = 0
        total_extracted_unknown = 0
        total_correct = 0
        image_accuracies = []
        
        for s in processed_samples:
            # Count samples with no questions
            if s.get("dsg_no_questions", False):
                samples_with_no_questions += 1
                continue
            
            # Count turns with no questions
            for turn in s.get("conversations", []):
                if turn.get("dsg_no_questions", False):
                    turns_with_no_questions += 1
            judgments = s.get("dsg_question_judgments", [])
            if judgments:
                sample_generated = len(judgments)
                sample_gt_unknown = sum(1 for j in judgments if j.get("skip_reason") == "gt_unknown")
                sample_extracted_unknown = sum(1 for j in judgments if j.get("skip_reason") == "extracted_unknown")
                # Evaluated = exclude both gt_unknown and extracted_unknown
                sample_evaluated = sample_generated - sample_gt_unknown - sample_extracted_unknown
                sample_correct = sum(1 for j in judgments if j.get("is_correct") == True)
                
                total_generated += sample_generated
                total_evaluated += sample_evaluated
                total_gt_unknown += sample_gt_unknown
                total_extracted_unknown += sample_extracted_unknown
                total_correct += sample_correct
                
                if sample_evaluated > 0:
                    image_accuracies.append(sample_correct / sample_evaluated)
        
        num_images_with_results = len(image_accuracies)
        
        # Response-level statistics - recalculate from question judgments
        response_accuracies = []
        response_total_generated = 0
        response_total_evaluated = 0
        response_total_correct = 0
        response_total_gt_unknown = 0
        response_total_extracted_unknown = 0
        
        for s in processed_samples:
            judgments = s.get("dsg_question_judgments", [])
            if judgments:
                # Group judgments by turn_index
                turn_judgments = {}
                for j in judgments:
                    turn_idx = j.get("turn_index", 0)
                    if turn_idx not in turn_judgments:
                        turn_judgments[turn_idx] = []
                    turn_judgments[turn_idx].append(j)
                
                for turn_idx, turn_js in turn_judgments.items():
                    turn_generated = len(turn_js)
                    turn_gt_unknown = sum(1 for j in turn_js if j.get("skip_reason") == "gt_unknown")
                    turn_extracted_unknown = sum(1 for j in turn_js if j.get("skip_reason") == "extracted_unknown")
                    # Evaluated = exclude both gt_unknown and extracted_unknown
                    turn_evaluated = turn_generated - turn_gt_unknown - turn_extracted_unknown
                    turn_correct = sum(1 for j in turn_js if j.get("is_correct") == True)
                    
                    response_total_generated += turn_generated
                    response_total_evaluated += turn_evaluated
                    response_total_gt_unknown += turn_gt_unknown
                    response_total_extracted_unknown += turn_extracted_unknown
                    response_total_correct += turn_correct
                    
                    if turn_evaluated > 0:
                        response_accuracies.append(turn_correct / turn_evaluated)
        
        num_responses = len(response_accuracies)
        
        print(f"\n{'='*60}")
        print(f"=== Overall Statistics ===")
        print(f"{'='*60}")
        
        print(f"\n[Image-level Statistics]")
        print(f"  Images with results: {num_images_with_results}")
        print(f"  Total generated questions: {total_generated}")
        print(f"  Extracted Unknown (skip): {total_extracted_unknown}")
        print(f"  Evaluated questions: {total_evaluated}")
        print(f"  Correct answers: {total_correct}")
        print(f"  GT Unknown (skip): {total_gt_unknown}")
        if total_evaluated > 0:
            print(f"  Accuracy: {total_correct / total_evaluated:.2%}")
        if image_accuracies:
            print(f"  Avg accuracy per image: {sum(image_accuracies) / len(image_accuracies):.2%}")
        
        print(f"\n[Response-level Statistics]")
        print(f"  Responses with results: {num_responses}")
        print(f"  Total generated questions: {response_total_generated}")
        print(f"  Extracted Unknown (skip): {response_total_extracted_unknown}")
        print(f"  Evaluated questions: {response_total_evaluated}")
        print(f"  Correct answers: {response_total_correct}")
        print(f"  GT Unknown (skip): {response_total_gt_unknown}")
        if response_total_evaluated > 0:
            print(f"  Accuracy: {response_total_correct / response_total_evaluated:.2%}")
        if response_accuracies:
            print(f"  Avg accuracy per response: {sum(response_accuracies) / len(response_accuracies):.2%}")