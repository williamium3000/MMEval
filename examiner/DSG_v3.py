#!/usr/bin/env python3
"""
DSG_v3.py: Extract answers from both Qwen-translated (from response) and fresh VLM answers.

Reads _pope_converted.json files that already have dsg_qa extracted.
For each question:
1. Get Qwen-translated answer from the response transcript (Qwen API, default port 8004)
2. Get fresh VLM answer by calling VLM with image + question (via infer/loader)
3. Optionally verify both against ground truth (--verify) and compare accuracy

Usage:
    python examiner/DSG_v3.py \
        --input_file work_dirs/vg/final_run_v18_gpt4o_completed/llava-1.5-7b-hf_pope_converted.json \
        --outfile work_dirs/vg/final_run_v18_gpt4o_completed/llava-1.5-7b-hf_with_both_answers.json \
        --vlm_model_path llava-hf/llava-1.5-7b-hf \
        --qwen_port 8004 \
        --verify
"""

import base64
import json
import os
import re
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from openai import OpenAI
from io import BytesIO
from urllib.parse import urlparse
import tqdm
import requests

API_RETRY_WAIT_SEC = 5
API_MAX_RETRIES = 5
API_TIMEOUT_SEC = 300

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Qwen API (local vLLM port 8004)
QWEN_BASE_URL = "http://localhost:8004/v1"
QWEN_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


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
1. If the object(s) mentioned in the question do NOT exist in the image annotation (not listed in objects), you MUST answer "No".
2. If the objects exist but a specific relationship is not mentioned in the annotation, answer "Unknown".
3. If the objects exist but a specific attribute is not mentioned, answer "Unknown".

Respond in JSON format as a list:
[
  {{"id": "<question_id>", "gt_answer": "Yes"/"No"/"Unknown", "reasoning": "brief explanation"}},
  ...
]

IMPORTANT: The "id" field must be the exact question ID (e.g., "0_1", "1_2") or the question index number.
"""

# Verify-only: one Qwen call per turn to map "response" -> Yes/No list for dsg_qa
RESPONSE_TO_YESNO_PROMPT = """Given a model's response and a list of yes/no questions, determine whether the response implies Yes, No, or Unknown for each question.

Response:
{response}

Questions (in order):
{questions}

For each question, output exactly one of: Yes, No, Unknown.
- Yes: the response clearly implies the statement in the question is true.
- No: the response clearly implies the statement is false or contradicts it.
- Unknown: the response does not contain enough information.

Respond with a JSON array of answers in the same order as the questions, e.g. ["Yes", "No", "Unknown", "Yes"].
Output only the JSON array, no other text."""


def format_conversation_as_transcript(conversation):
    transcript_parts = []
    for i, turn in enumerate(conversation):
        role = turn.get("role", "")
        if not role:
            role = "qa_pair" if ("prompt" in turn and "response" in turn) else "unknown"
        if role == "qa_pair" or ("prompt" in turn and "response" in turn):
            transcript_parts.append(f"[Turn {i+1}] QUESTION: {turn.get('prompt', '')}")
            transcript_parts.append(f"[Turn {i+1}] ANSWER: {turn.get('response', '')}")
        else:
            content = turn.get("content") or turn.get("response", "")
            transcript_parts.append(f"[Turn {i+1}] {role.upper()}: {content}")
    return "\n\n".join(transcript_parts)


def format_vg_annotation(sample):
    """Format Visual Genome style annotation as readable string."""
    annotation_parts = []
    if "metadata" in sample:
        sample = sample["metadata"]
    sg_data = sample.get("sg", sample)
    objects_data = sg_data.get("objects", sample.get("objects"))
    if objects_data:
        names = []
        if isinstance(objects_data, dict):
            for obj in objects_data.values():
                n = obj.get("names", [])
                if n:
                    names.append(n[0])
        elif isinstance(objects_data, list):
            for obj in objects_data:
                n = obj.get("names", [])
                if n:
                    names.append(n[0])
        if names:
            annotation_parts.append(f"Objects in image: {', '.join(set(names))}")
    rel_data = sg_data.get("relationships", sample.get("relationships"))
    if rel_data:
        rels = []
        if isinstance(rel_data, dict):
            for r in rel_data.values():
                s, o, p = r.get("subject"), r.get("object"), r.get("predicate", "")
                if s and o and p:
                    rels.append(f"{s} {p} {o}")
        elif isinstance(rel_data, list):
            for r in rel_data:
                s, o, p = r.get("subject"), r.get("object"), r.get("predicate", "")
                if s and o and p:
                    rels.append(f"{s} {p} {o}")
        if rels:
            annotation_parts.append("Relationships:\n  " + "\n  ".join(rels))
    return "\n\n".join(annotation_parts) if annotation_parts else "No annotation available"


def extract_answers_from_transcript(questions_dict, transcript, get_response):
    questions_str = "\n".join(f"{qid}. {q}" for qid, q in questions_dict.items())
    prompt = BATCH_ANSWER_EXTRACTION_PROMPT.format(transcript=transcript, questions=questions_str)
    for attempt in range(API_MAX_RETRIES):
        try:
            response = get_response(prompt)
            break
        except Exception as e:
            if attempt < API_MAX_RETRIES - 1:
                time.sleep(API_RETRY_WAIT_SEC)
            else:
                raise
    results = {}
    try:
        m = re.search(r'\[[\s\S]*\]', response)
        if m:
            for item in json.loads(m.group()):
                qid = item.get("id")
                if qid is not None:
                    if isinstance(qid, str) and qid.isdigit():
                        qid = int(qid)
                    results[qid] = {
                        "has_answer": item.get("has_answer", False),
                        "answer": item.get("answer", "Unknown"),
                        "evidence": item.get("evidence", ""),
                    }
    except (json.JSONDecodeError, AttributeError):
        pass
    for qid in questions_dict:
        if qid not in results:
            results[qid] = {"has_answer": False, "answer": "Unknown", "evidence": ""}
    return results


def verify_ground_truth(questions_dict, annotation_str, get_response):
    questions_str = "\n".join(f"{qid}. {q}" for qid, q in questions_dict.items())
    prompt = GROUND_TRUTH_VERIFICATION_PROMPT.format(annotation=annotation_str, questions=questions_str)
    for attempt in range(API_MAX_RETRIES):
        try:
            response = get_response(prompt)
            break
        except Exception as e:
            if attempt < API_MAX_RETRIES - 1:
                time.sleep(API_RETRY_WAIT_SEC)
            else:
                raise
    results = {}
    try:
        m = re.search(r'\[[\s\S]*\]', response)
        if m:
            for item in json.loads(m.group()):
                qid = item.get("id")
                if qid is not None:
                    results[qid] = {"gt_answer": item.get("gt_answer", "Unknown"), "reasoning": item.get("reasoning", "")}
    except (json.JSONDecodeError, AttributeError):
        pass
    for qid in questions_dict:
        if qid not in results:
            results[qid] = {"gt_answer": "Unknown", "reasoning": ""}
    return results


def normalize_answer(a):
    if not a:
        return "unknown"
    a = a.lower().strip()
    if "unknown" in a or "unclear" in a or "n/a" in a:
        return "unknown"
    if "yes" in a and "no" not in a:
        return "yes"
    if "no" in a:
        return "no"
    return "unknown"


def compute_accuracy(extracted, gt):
    en = normalize_answer(extracted)
    gn = normalize_answer(gt)
    if gn == "unknown":
        return {"is_correct": False, "skip_reason": "gt_unknown"}
    if en == "unknown":
        return {"is_correct": None, "skip_reason": "extracted_unknown"}
    return {"is_correct": (en == gn), "skip_reason": None}


def batch_accuracy_stats(samples_or_batch, batch_idx=None):
    """Compute accuracy and skip-reason counts over a list of samples. Returns dict for logging."""
    qw_ok, qw_ev, vlm_ok, vlm_ev = 0, 0, 0, 0
    gt_unk, qw_unk, vlm_unk = 0, 0, 0
    for s in samples_or_batch:
        for t in s.get("conversations", []):
            for qa in t.get("dsg_qa", []):
                if "qwen_answer" not in qa:
                    continue
                gt = qa.get("gt_answer", "Unknown")
                if gt == "Unknown" or normalize_answer(gt) == "unknown":
                    gt_unk += 1
                    continue
                # Qwen
                sr = qa.get("qwen_skip_reason")
                if not sr:
                    qw_ev += 1
                    if qa.get("qwen_is_correct"):
                        qw_ok += 1
                elif sr == "extracted_unknown":
                    qw_unk += 1
                # VLM
                sr = qa.get("vlm_skip_reason")
                if not sr:
                    vlm_ev += 1
                    if qa.get("vlm_is_correct"):
                        vlm_ok += 1
                elif sr == "extracted_unknown":
                    vlm_unk += 1
    out = {
        "qw_ok": qw_ok, "qw_ev": qw_ev,
        "vlm_ok": vlm_ok, "vlm_ev": vlm_ev,
        "gt_unk": gt_unk, "qw_unk": qw_unk, "vlm_unk": vlm_unk,
    }
    if batch_idx is not None:
        out["batch_idx"] = batch_idx
    return out


def log_batch_acc(stats, prefix="Batch"):
    """Print one-line accuracy and skip-reason stats (for per-batch logging). Use tqdm.write so progress bar doesn't overwrite."""
    b = stats.get("batch_idx", "")
    qw = f"{stats['qw_ok']}/{stats['qw_ev']}" if stats["qw_ev"] else "0/0"
    vl = f"{stats['vlm_ok']}/{stats['vlm_ev']}" if stats["vlm_ev"] else "0/0"
    qw_pct = f"{100 * stats['qw_ok'] / stats['qw_ev']:.1f}%" if stats["qw_ev"] else "N/A"
    vl_pct = f"{100 * stats['vlm_ok'] / stats['vlm_ev']:.1f}%" if stats["vlm_ev"] else "N/A"
    sk = f"skip gt_unk={stats['gt_unk']} qw_unk={stats['qw_unk']} vl_unk={stats['vlm_unk']}"
    msg = f"  [{prefix} {b}] Qwen: {qw} = {qw_pct}  VLM: {vl} = {vl_pct}  | {sk}"
    tqdm.tqdm.write(msg)


def _effective_ref_verify_only(qa):
    """Reference for verify_only: gt_answer if not Unknown, else normalized qa['answer']."""
    gt = qa.get("gt_answer")
    if gt is not None and gt != "Unknown" and normalize_answer(gt) != "unknown":
        return gt
    return normalize_qa_answer_to_yes_no(qa.get("answer", ""))


def batch_accuracy_stats_verify_only(samples_or_batch, batch_idx=None):
    """Accuracy for verify_only: dynamic/single vs effective ref (gt_answer or answer)."""
    dyn_ok, dyn_ev, single_ok, single_ev = 0, 0, 0, 0
    ref_unk = 0
    for s in samples_or_batch:
        for t in s.get("conversations", []):
            for qa in t.get("dsg_qa", []):
                ref = _effective_ref_verify_only(qa)
                if ref == "Unknown" or normalize_answer(ref) == "unknown":
                    ref_unk += 1
                    continue
                if "dynamic_response" in qa:
                    dyn_ev += 1
                    if compute_accuracy(qa["dynamic_response"], ref)["is_correct"]:
                        dyn_ok += 1
                if "single_response" in qa:
                    single_ev += 1
                    if compute_accuracy(qa["single_response"], ref)["is_correct"]:
                        single_ok += 1
    out = {"dyn_ok": dyn_ok, "dyn_ev": dyn_ev, "single_ok": single_ok, "single_ev": single_ev, "gt_unk": ref_unk}
    if batch_idx is not None:
        out["batch_idx"] = batch_idx
    return out


def log_batch_acc_verify_only(stats, prefix="Batch"):
    """Log dynamic (Qwen) and single (VLM) accuracy for verify_only."""
    b = stats.get("batch_idx", "")
    dyn = f"{stats['dyn_ok']}/{stats['dyn_ev']}" if stats["dyn_ev"] else "0/0"
    single = f"{stats['single_ok']}/{stats['single_ev']}" if stats["single_ev"] else "0/0"
    dyn_pct = f"{100 * stats['dyn_ok'] / stats['dyn_ev']:.1f}%" if stats["dyn_ev"] else "N/A"
    single_pct = f"{100 * stats['single_ok'] / stats['single_ev']:.1f}%" if stats["single_ev"] else "N/A"
    msg = f"  [{prefix} {b}] dynamic (response): {dyn} = {dyn_pct}  |  single (VLM): {single} = {single_pct}  |  skip ref_unk={stats['gt_unk']}"
    tqdm.tqdm.write(msg)


def normalize_vlm_answer(text):
    t = text.lower().strip()
    if "yes" in t and "no" not in t:
        return "Yes"
    if "no" in t:
        return "No"
    return "Unknown"


def normalize_qa_answer_to_yes_no(text):
    """Normalize existing qa['answer'] (from transcript) to Yes/No/Unknown."""
    if not text or not str(text).strip():
        return "Unknown"
    t = str(text).lower().strip()
    if "yes" in t and "no" not in t:
        return "Yes"
    if "no" in t:
        return "No"
    return "Unknown"


def prepare_image_for_api(image_path_or_url):
    """Prepare image as base64 for OpenAI-compatible vision API."""
    if urlparse(image_path_or_url).scheme in ("http", "https"):
        r = requests.get(image_path_or_url, timeout=30)
        r.raise_for_status()
        image_bytes = r.content
        mime = "image/jpeg"
    else:
        with open(image_path_or_url, "rb") as f:
            image_bytes = f.read()
        ext = os.path.splitext(image_path_or_url)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}.get(ext, "image/jpeg")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def call_vlm_api(api_url: str, image_path: str, question: str, model_name: str) -> str:
    """Call VLM via OpenAI-compatible API (e.g. vLLM) with image + question."""
    image_content = prepare_image_for_api(image_path)
    messages = [{"role": "user", "content": [image_content, {"type": "text", "text": question}]}]
    payload = {"model": model_name, "messages": messages, "max_tokens": 512, "temperature": 0.0}
    base = api_url.rstrip("/").replace("/chat/completions", "")
    if not base.endswith("/v1"):
        base = base + "/v1"
    url = base + "/chat/completions"
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    out = r.json()
    if out.get("choices") and len(out["choices"]) > 0:
        return (out["choices"][0].get("message", {}).get("content") or "").strip()
    return ""


def call_vlm_api_multi(api_url: str, image_path_or_url: str, questions: list, model_name: str) -> str:
    """One VLM call: image + list of questions, ask for one Yes/No per line. Returns raw text."""
    questions_blob = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    n = len(questions)
    prompt = f"""Look at the image and answer each question with ONLY Yes or No. Give exactly {n} answers, one per line, in the same order as the questions. Write nothing else.

Questions:
{questions_blob}

Your {n} answers (one word per line):"""
    image_content = prepare_image_for_api(image_path_or_url)
    messages = [{"role": "user", "content": [image_content, {"type": "text", "text": prompt}]}]
    payload = {"model": model_name, "messages": messages, "max_tokens": 512, "temperature": 0.0}
    base = api_url.rstrip("/").replace("/chat/completions", "")
    if not base.endswith("/v1"):
        base = base + "/v1"
    url = base + "/chat/completions"
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    out = r.json()
    if out.get("choices") and len(out["choices"]) > 0:
        return (out["choices"][0].get("message", {}).get("content") or "").strip()
    return ""


def parse_yesno_list(raw: str, n_expected: int) -> list:
    """Parse VLM raw output into list of Yes/No/Unknown. Prefer Yes/No over Unknown when model returns a short reply (e.g. single 'Yes')."""
    if not raw or not raw.strip():
        return ["Unknown"] * n_expected
    text = raw.strip()
    out = []

    # 1) Try line-by-line first (one Yes/No per line)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for ln in lines:
        ln_clean = re.sub(r"^\s*\d+[\.\)]\s*", "", ln.lower(), flags=re.I).strip()
        if ln_clean in ("yes", "y"):
            out.append("Yes")
        elif ln_clean in ("no", "n"):
            out.append("No")
        elif "yes" in ln_clean and "no" not in ln_clean:
            out.append("Yes")
        elif "no" in ln_clean:
            out.append("No")
        else:
            out.append("Unknown")

    # 2) If we got too few, extract all Yes/No tokens from full text (handles "Yes No Yes" or "Yes, No, Yes")
    if len(out) < n_expected:
        # Find all standalone yes/no in order (case-insensitive)
        tokens = re.findall(r"\b(yes|no|y|n)\b", text, re.I)
        extracted = []
        for t in tokens:
            if t.lower() in ("yes", "y"):
                extracted.append("Yes")
            else:
                extracted.append("No")
        if len(extracted) > len(out):
            out = extracted
    # 3) If still short and we have at least one Yes/No, replicate last to fill (avoids Unknown when model said e.g. single "Yes")
    if len(out) < n_expected and len(out) > 0:
        last = out[-1] if out[-1] in ("Yes", "No") else "Unknown"
        while len(out) < n_expected:
            out.append(last)
    while len(out) < n_expected:
        out.append("Unknown")
    return out[:n_expected]


def qwen_response_to_yesno_list(response: str, questions: list, get_qwen) -> list:
    """One Qwen call: does 'response' imply Yes/No for each question? Returns list of Yes/No/Unknown."""
    if not response or not questions:
        return ["Unknown"] * len(questions)
    questions_str = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    prompt = RESPONSE_TO_YESNO_PROMPT.format(response=response, questions=questions_str)
    for attempt in range(API_MAX_RETRIES):
        try:
            text = get_qwen(prompt)
            break
        except Exception as e:
            if attempt < API_MAX_RETRIES - 1:
                time.sleep(API_RETRY_WAIT_SEC)
            else:
                return ["Unknown"] * len(questions)
    out = []
    try:
        m = re.search(r"\[[\s\S]*?\]", text)
        if m:
            arr = json.loads(m.group())
            for i, item in enumerate(arr):
                if i >= len(questions):
                    break
                a = item if isinstance(item, str) else item.get("answer", item)
                if isinstance(a, str):
                    a = a.strip()
                    if "yes" in a.lower() and "no" not in a.lower():
                        out.append("Yes")
                    elif "no" in a.lower():
                        out.append("No")
                    else:
                        out.append("Unknown")
                else:
                    out.append("Unknown")
    except (json.JSONDecodeError, TypeError):
        pass
    while len(out) < len(questions):
        out.append("Unknown")
    return out[:len(questions)]


def main():
    parser = argparse.ArgumentParser(description="DSG v3: Qwen-translated + fresh VLM answers")
    parser.add_argument("--input_file", type=str, required=True, help="Input _pope_converted.json")
    parser.add_argument("--outfile", type=str, required=True, help="Output JSON")
    parser.add_argument("--vlm_model_path", type=str, default=None, help="VLM model path when loading locally (e.g. llava-hf/llava-1.5-7b-hf)")
    parser.add_argument("--vlm_model_base", type=str, default=None)
    parser.add_argument("--vlm_api_url", type=str, default=None, help="VLM API URL (e.g. http://localhost:8005/v1). If set, use API instead of loading model.")
    parser.add_argument("--vlm_api_model", type=str, default=None, help="Model name for VLM API (e.g. llava-hf/llava-1.5-7b-hf). Required if --vlm_api_url is set.")
    parser.add_argument("--qwen_port", type=int, default=8004)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--sample_num", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="If outfile exists, load it and skip samples that already have results (verify_only: dynamic_response/single_response)")
    parser.add_argument("--verify", action="store_true", help="Verify both answers against ground truth")
    parser.add_argument("--verify_only", action="store_true", help="Lighter path when input already has questions: use existing qa['answer'], VLM only, Qwen for GT only (no full transcript/scene graph)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--vlm_workers", type=int, default=36, help="Max parallel VLM API calls per batch")
    parser.add_argument("--qwen_workers", type=int, default=18, help="Max parallel Qwen API calls per batch (all samples in batch queried concurrently)")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.qwen_port}/v1"
    qwen_client = OpenAI(api_key="dummy-key", base_url=base_url, timeout=API_TIMEOUT_SEC)

    def get_qwen(prompt):
        return qwen_client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        ).choices[0].message.content

    use_vlm_api = bool(args.vlm_api_url)
    vlm_eval_fn = None
    vlm_api_model = None
    if use_vlm_api:
        vlm_api_model = args.vlm_api_model or args.vlm_model_path
        if not vlm_api_model:
            raise SystemExit("When --vlm_api_url is set, provide --vlm_api_model or --vlm_model_path.")
        print(f"Using VLM via API: {args.vlm_api_url} model={vlm_api_model}")
    else:
        if not args.vlm_model_path:
            raise SystemExit("Provide either --vlm_model_path (load locally) or --vlm_api_url (use API).")
        print(f"Loading VLM locally: {args.vlm_model_path}")
        loader_args = type("Args", (), {"model_path": args.vlm_model_path, "model_base": args.vlm_model_base})()
        from infer.loader import load_model
        vlm_eval_fn = load_model(loader_args)

    # Load canonical list from input file
    with open(args.input_file, "r") as f:
        input_samples = json.load(f)

    def _sample_key_list(sample_list):
        """Build (image_id, occurrence) for each sample so we can match across input/outfile with gaps."""
        from collections import defaultdict
        occ = defaultdict(int)
        keys = []
        for s in sample_list:
            im = s.get("image_id", "unknown")
            keys.append((im, occ[im]))
            occ[im] += 1
        return keys

    def _is_complete_verify_only(sample):
        for turn in sample.get("conversations", []):
            for qa in turn.get("dsg_qa", []):
                if "dynamic_response" not in qa or "single_response" not in qa:
                    return False
        return True

    def _is_complete_full(sample):
        for turn in sample.get("conversations", []):
            for qa in turn.get("dsg_qa", []):
                if "qwen_answer" not in qa:
                    return False
        return True

    if getattr(args, "resume", False) and os.path.isfile(args.outfile):
        with open(args.outfile, "r") as f:
            out_samples = json.load(f)
        input_keys = _sample_key_list(input_samples)
        out_keys = _sample_key_list(out_samples)
        is_complete = _is_complete_verify_only if args.verify_only else _is_complete_full
        completed_keys = set()
        out_by_key = {}
        for j, s in enumerate(out_samples):
            k = out_keys[j] if j < len(out_keys) else (s.get("image_id", "unknown"), j)
            out_by_key[k] = s
            if is_complete(s):
                completed_keys.add(k)
        # Merge: full list aligned to input; use outfile sample where key is complete, else input (to process)
        samples = [None] * len(input_samples)
        to_process_indices = []
        for i in range(len(input_samples)):
            key = input_keys[i]
            if key in completed_keys and key in out_by_key:
                samples[i] = out_by_key[key]
            else:
                samples[i] = input_samples[i]
                to_process_indices.append(i)
        samples_to_process = [samples[i] for i in to_process_indices]
        n_complete = len(input_samples) - len(to_process_indices)
        print(f"Resume: loaded {args.outfile} (n={len(out_samples)}), input n={len(input_samples)}; "
              f"complete by (image_id,occ): {n_complete}, to process: {len(samples_to_process)}")
    else:
        samples = input_samples
        args.start_idx = int(args.start_idx)
        end = args.start_idx + args.sample_num if args.sample_num else len(samples)
        samples_to_process = samples[args.start_idx:end]
        print(f"Processing {len(samples_to_process)} samples (start_idx={args.start_idx})")

    if not samples_to_process:
        print("Nothing to process (all samples already complete or empty range). Exiting.")
        return

    def get_image_path(sample):
        """Resolve image path: some datasets use 'image', others 'url' or 'image_path'."""
        return sample.get("image") or sample.get("url") or sample.get("image_path")

    def do_one_sample_qwen(sample):
        """Run Qwen extraction (+ optional GT verify) for one sample. Returns (sample, qwen_results, gt_results) or None."""
        all_questions = {}
        for ti, turn in enumerate(sample.get("conversations", [])):
            for qa in turn.get("dsg_qa", []):
                qid = qa.get("index")
                comp = f"{ti}_{qid}"
                all_questions[comp] = qa.get("question", "")
        if not all_questions:
            return None
        transcript = format_conversation_as_transcript(sample.get("conversations", []))
        qwen_results = extract_answers_from_transcript(all_questions, transcript, get_qwen)
        gt_results = {}
        if args.verify:
            ann = format_vg_annotation(sample)
            gt_results = verify_ground_truth(all_questions, ann, get_qwen)
        return (sample, qwen_results, gt_results)

    def do_one_vlm(item):
        """One VLM call; item is (img_path, question, qa). Returns (qa, raw)."""
        img_path, question, qa = item
        if use_vlm_api:
            raw = call_vlm_api(args.vlm_api_url, img_path, question, vlm_api_model)
            return qa, raw
        img = Image.open(img_path).convert("RGB")
        raw = vlm_eval_fn(img, question)
        return qa, (raw[0] if isinstance(raw, tuple) else raw)

    def do_one_sample_gt_only(sample):
        """Verify-only: get GT only (no transcript extraction). Returns (sample, gt_results) or None."""
        all_questions = {}
        for ti, turn in enumerate(sample.get("conversations", [])):
            for qa in turn.get("dsg_qa", []):
                qid = qa.get("index")
                comp = f"{ti}_{qid}"
                all_questions[comp] = qa.get("question", "")
        if not all_questions:
            return None
        ann = format_vg_annotation(sample)
        gt_results = verify_ground_truth(all_questions, ann, get_qwen)
        return (sample, gt_results)

    if args.verify_only:
        # Verify-only: 2 LLM calls per turn — (1) Qwen: response -> Yes/No list for dsg_qa; (2) VLM: image + questions -> one answer list
        print("Verify-only mode: 1 Qwen call per turn (response -> dynamic_response), 1 VLM call per turn (image + questions -> single_response)")
        for batch_start in tqdm.tqdm(range(0, len(samples_to_process), args.batch_size), desc="Batches"):
            batch = samples_to_process[batch_start : batch_start + args.batch_size]
            # 1) GT only (one Qwen call per sample)
            with ThreadPoolExecutor(max_workers=args.qwen_workers) as ex:
                gt_futures = [ex.submit(do_one_sample_gt_only, sample) for sample in batch]
                gt_outputs = [f.result() for f in gt_futures]
            # 2) Per sample: assign GT; per turn with response+dsg_qa: 1 Qwen (response->list), 1 VLM (image+questions->list)
            for out in gt_outputs:
                if out is None:
                    continue
                sample, gt_results = out
                image_path = get_image_path(sample)
                if not image_path:
                    continue
                for ti, turn in enumerate(sample.get("conversations", [])):
                    response = turn.get("response", "")
                    qa_list = turn.get("dsg_qa", [])
                    if not response or not qa_list:
                        continue
                    questions = [qa.get("question", "") for qa in qa_list]
                    for qa in qa_list:
                        qid = qa.get("index")
                        comp = f"{ti}_{qid}"
                        gt_info = gt_results.get(comp) or gt_results.get(qid, {})
                        gt_val = gt_info.get("gt_answer", "Unknown")
                        if gt_val != "Unknown" and normalize_answer(gt_val) != "unknown":
                            qa["gt_answer"] = gt_val
                            qa["gt_reasoning"] = gt_info.get("reasoning", "")
                        # when gt is Unknown we omit gt_answer; comparison will use qa["answer"] instead
                    # (1) Qwen: one call per turn — does "response" imply Yes/No for each question?
                    dynamic_list = qwen_response_to_yesno_list(response, questions, get_qwen)
                    for i, qa in enumerate(qa_list):
                        qa["dynamic_response"] = dynamic_list[i] if i < len(dynamic_list) else "Unknown"
                    # (2) VLM: one call per turn — image + list of questions, parse list of Yes/No
                    try:
                        single_raw = call_vlm_api_multi(args.vlm_api_url, image_path, questions, vlm_api_model)
                        single_list = parse_yesno_list(single_raw, len(questions))
                        for i, qa in enumerate(qa_list):
                            qa["single_response"] = single_list[i] if i < len(single_list) else "Unknown"
                        for qa in qa_list:
                            qa["single_response_raw"] = single_raw
                    except Exception as e:
                        if args.verbose:
                            print(f"VLM error: {e}")
                        for qa in qa_list:
                            qa["single_response"] = "Unknown"
                            qa["single_response_raw"] = ""
                # Ensure any turn without response/dsg_qa still has gt_answer from gt_results
                for ti, turn in enumerate(sample.get("conversations", [])):
                    for qa in turn.get("dsg_qa", []):
                        comp = f"{ti}_{qa.get('index')}"
                        if "gt_answer" not in qa:
                            gt_info = gt_results.get(comp) or gt_results.get(qa.get("index"), {})
                            g = gt_info.get("gt_answer", "Unknown")
                            if g != "Unknown" and normalize_answer(g) != "unknown":
                                qa["gt_answer"] = g
                                qa["gt_reasoning"] = gt_info.get("reasoning", "")
            # 3) Compare dynamic_response and single_response to effective ref (gt_answer or answer)
            for sample in batch:
                for turn in sample.get("conversations", []):
                    for qa in turn.get("dsg_qa", []):
                        ref = _effective_ref_verify_only(qa)
                        if ref == "Unknown" or normalize_answer(ref) == "unknown":
                            continue
                        if "dynamic_response" in qa:
                            acc = compute_accuracy(qa["dynamic_response"], ref)
                            qa["dynamic_is_correct"] = acc["is_correct"]
                        if "single_response" in qa:
                            acc = compute_accuracy(qa["single_response"], ref)
                            qa["single_is_correct"] = acc["is_correct"]
            batch_idx = batch_start // args.batch_size
            st = batch_accuracy_stats_verify_only(batch, batch_idx=batch_idx)
            log_batch_acc_verify_only(st, prefix="Batch")
            with open(args.outfile, "w") as f:
                json.dump(samples, f, indent=2)
    else:
        # Full path: Qwen extracts from transcript + GT, VLM, then compare
        for batch_start in tqdm.tqdm(range(0, len(samples_to_process), args.batch_size), desc="Batches"):
            batch = samples_to_process[batch_start : batch_start + args.batch_size]

            # 1) Run all Qwen (+ GT) calls for this batch in parallel
            with ThreadPoolExecutor(max_workers=args.qwen_workers) as ex:
                qwen_futures = [ex.submit(do_one_sample_qwen, sample) for sample in batch]
                qwen_outputs = [f.result() for f in qwen_futures]

            # 2) Assign Qwen results and build batch-wide VLM task list (img_path, question, qa)
            batch_vlm_tasks = []
            for out in qwen_outputs:
                if out is None:
                    continue
                sample, qwen_results, gt_results = out
                image_path = get_image_path(sample)
                if not image_path:
                    continue
                for ti, turn in enumerate(sample.get("conversations", [])):
                    for qa in turn.get("dsg_qa", []):
                        qid = qa.get("index")
                        comp = f"{ti}_{qid}"
                        r = qwen_results.get(comp) or qwen_results.get(qid, {})
                        qa["qwen_answer"] = r.get("answer", "Unknown")
                        qa["qwen_has_answer"] = r.get("has_answer", False)
                        qa["qwen_evidence"] = r.get("evidence", "")
                        batch_vlm_tasks.append((image_path, qa.get("question", ""), qa))
                if args.verify:
                    for ti, turn in enumerate(sample.get("conversations", [])):
                        for qa in turn.get("dsg_qa", []):
                            qid = qa.get("index")
                            comp = f"{ti}_{qid}"
                            gt_info = gt_results.get(comp) or gt_results.get(qid, {})
                            qa["gt_answer"] = gt_info.get("gt_answer", "Unknown")
                            qa["gt_reasoning"] = gt_info.get("reasoning", "")

            # 3) Run all VLM calls for the whole batch in parallel
            with ThreadPoolExecutor(max_workers=args.vlm_workers) as ex:
                future_to_qa = {ex.submit(do_one_vlm, t): t[2] for t in batch_vlm_tasks}
                for fut in as_completed(future_to_qa):
                    qa = future_to_qa[fut]
                    try:
                        _, raw = fut.result()
                        qa["vlm_answer_raw"] = raw
                        qa["vlm_answer"] = normalize_vlm_answer(raw)
                    except Exception as e:
                        if args.verbose:
                            print(f"VLM error: {e}")
                        qa["vlm_answer"] = "Unknown"
                        qa["vlm_answer_raw"] = ""

            # 4) Compute accuracy (verify) only for qa we actually processed (have qwen_answer)
            if args.verify:
                for sample in batch:
                    for turn in sample.get("conversations", []):
                        for qa in turn.get("dsg_qa", []):
                            if "qwen_answer" not in qa:
                                continue
                            gt_answer = qa.get("gt_answer", "Unknown")
                            qw_acc = compute_accuracy(qa["qwen_answer"], gt_answer)
                            vl_acc = compute_accuracy(qa.get("vlm_answer", "Unknown"), gt_answer)
                            qa["qwen_is_correct"] = qw_acc["is_correct"]
                            qa["qwen_skip_reason"] = qw_acc.get("skip_reason")
                            qa["vlm_is_correct"] = vl_acc["is_correct"]
                            qa["vlm_skip_reason"] = vl_acc.get("skip_reason")

            # Per-batch accuracy (debug / early signal)
            batch_idx = batch_start // args.batch_size
            st = batch_accuracy_stats(batch, batch_idx=batch_idx)
            log_batch_acc(st, prefix="Batch")

            with open(args.outfile, "w") as f:
                json.dump(samples, f, indent=2)

    print(f"Results saved to {args.outfile}")

    if args.verify or args.verify_only:
        if args.verify_only:
            dyn_ok, dyn_ev, single_ok, single_ev = 0, 0, 0, 0
            for s in samples_to_process:
                for t in s.get("conversations", []):
                    for qa in t.get("dsg_qa", []):
                        # Count only where we had a valid ref (gt or answer) and set is_correct
                        if "dynamic_is_correct" in qa:
                            dyn_ev += 1
                            if qa.get("dynamic_is_correct"):
                                dyn_ok += 1
                        if "single_is_correct" in qa:
                            single_ev += 1
                            if qa.get("single_is_correct"):
                                single_ok += 1
            print(f"\n=== Accuracy (verify_only, ref=gt or answer) ===")
            print(f"dynamic (response→Qwen): {dyn_ok}/{dyn_ev} = {dyn_ok/dyn_ev:.2%}" if dyn_ev else "dynamic: N/A")
            print(f"single (image+VLM):      {single_ok}/{single_ev} = {single_ok/single_ev:.2%}" if single_ev else "single: N/A")
        else:
            qw_ok, qw_ev, vlm_ok, vlm_ev = 0, 0, 0, 0
            for s in samples_to_process:
                for t in s.get("conversations", []):
                    for qa in t.get("dsg_qa", []):
                        if qa.get("gt_answer") == "Unknown":
                            continue
                        if not qa.get("qwen_skip_reason"):
                            qw_ev += 1
                            if qa.get("qwen_is_correct"):
                                qw_ok += 1
                        if not qa.get("vlm_skip_reason"):
                            vlm_ev += 1
                            if qa.get("vlm_is_correct"):
                                vlm_ok += 1
            print(f"\n=== Accuracy (evaluated only) ===")
            print(f"Qwen-translated: {qw_ok}/{qw_ev} = {qw_ok/qw_ev:.2%}" if qw_ev else "Qwen-translated: N/A")
            print(f"Fresh VLM:       {vlm_ok}/{vlm_ev} = {vlm_ok/vlm_ev:.2%}" if vlm_ev else "Fresh VLM: N/A")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Exiting (Ctrl+C). Partial results may be in outfile.")
        sys.exit(130)
