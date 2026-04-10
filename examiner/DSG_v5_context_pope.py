#!/usr/bin/env python3
"""
DSG_v5_context_pope.py: Convert conv_script to output with dsg_qa (object-existence questions).

Uses exactly 2 LLM calls per conversation:
1) Given context and goal from conv metadata, ask LLM for a pool of N_CANDIDATES (50) objects.
2) Given the scene graph's existing object list, ask LLM whether each candidate exists
   (synonym/parent class count as exist).

Pool size: first LLM is asked for N_CANDIDATES (50) objects. Output count = number of rounds in the conv (n_rounds).
Final composition per sample: 1/2 yes, 1/4 random from all no, 1/4 random from no-and-co-occur
(where co-occur = non-existing candidates that frequently co-occur with the existing list; from 2nd LLM).

Output: one dsg_qa item per turn (length n_rounds), original-format keys:
  {"index": i, "question": "is there a <object> in the image?", "answer": "yes"|"no"}
"""

import json
import os
import random
import re
import sys
import time
import argparse
from functools import partial

import tqdm

# Pool size: how many candidates to ask the first LLM for (output count = n_rounds per sample)
N_CANDIDATES = 50
from openai import OpenAI

API_RETRY_WAIT_SEC = 5
API_MAX_RETRIES = 5
API_TIMEOUT_SEC = 300

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_project_root, ".env"))
except ImportError:
    pass

REMOTE_API_URL = os.getenv("REMOTE_API_URL", "")
if REMOTE_API_URL:
    REMOTE_BASE_URL = REMOTE_API_URL.rstrip("/chat/completions").rstrip("/v1")
    if not REMOTE_BASE_URL.endswith("/v1"):
        REMOTE_BASE_URL = REMOTE_BASE_URL + "/v1"
else:
    REMOTE_BASE_URL = ""
REMOTE_API_KEY = os.getenv("REMOTE_API_KEY", "")
REMOTE_MODEL = os.getenv("REMOTE_API_MODEL", "Qwen3-30B-A3B-Instruct-2507")

LOCAL_BASE_URL = "http://109.61.17.115:30004/v1"
LOCAL_API_KEY = ""
LOCAL_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

if REMOTE_BASE_URL:
    client = OpenAI(api_key=REMOTE_API_KEY, base_url=REMOTE_BASE_URL, timeout=API_TIMEOUT_SEC)
else:
    client = None


OBJECTS_FROM_CONTEXT_PROMPT = """You are given a short scenario description (context and goal) for an image-based conversation.

Context (background): {background}

Goal: {goal}

List exactly {n} objects that might appear in this scenario. Include a mix: some that likely ARE in the scene and some that might NOT be (plausible but absent). Use concrete, visible things (e.g. "parking meter", "clock", "tree", "car", "bus stop", "bench"). Return ONLY a JSON array of exactly {n} strings, e.g.:
["object1", "object2", "object3", ...]

Output (JSON array only, no other text):"""


EXIST_AND_COOCCUR_PROMPT = """You are given:
1) The list of objects that actually exist in the image (from a scene graph): {existing_list}
2) A list of {n} candidate objects (with 0-based indices): {candidates_list}

Do two things:

A) For each candidate, say if it EXISTS in the image. Count synonyms and parent/superclass as existing (e.g. "vehicle" exists if "car" is in the list). Reply with a JSON array of exactly {n} answers in order: each "yes" or "no".
   Example: ["yes", "no", "yes", "no", ...]

B) Among the candidates that do NOT exist (answer "no"), which ones frequently co-occur with the existing objects in real-world scenes? (e.g. "bus stop" often appears with "street", "bench" with "park"; these are plausible distractors.) Reply with a JSON array of the 0-based indices of those non-existing candidates that co-occur with the existing list. It can be empty if none apply.
   Example: [3, 7, 12]

Return exactly two lines:
Line 1 (exists): ["yes","no",...]
Line 2 (co_occur_no_indices): [1, 5, 9]

Output:"""


def get_llm_response(prompt, model=None, temperature=0, max_tokens=1024):
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
            return completion.choices[0].message.content
        except Exception as e:
            last_err = e
            if attempt < API_MAX_RETRIES - 1:
                time.sleep(API_RETRY_WAIT_SEC)
            else:
                raise last_err


def extract_object_list_from_sg(sample):
    """Build list of object names (and optional synset stems) from scene graph. Returns (names_set, display_list)."""
    names_set = set()
    display_list = []
    sg = sample.get("sg") or sample.get("metadata") or {}
    objects = sg.get("objects")
    if not objects:
        return names_set, display_list
    items = objects.values() if isinstance(objects, dict) else objects
    for obj in items:
        if not isinstance(obj, dict):
            continue
        for name in obj.get("names") or []:
            if name and isinstance(name, str):
                n = name.strip().lower()
                if n:
                    names_set.add(n)
                    display_list.append(name.strip())
        for syn in obj.get("synsets") or []:
            if syn and isinstance(syn, str):
                # e.g. "clock.n.01" -> "clock"
                stem = syn.split(".")[0].replace("_", " ").strip().lower()
                if stem:
                    names_set.add(stem)
    return names_set, display_list


def parse_json_array(raw, n_expected, default_item=None):
    """Parse LLM output as JSON array; return list of length n_expected."""
    if default_item is None:
        default_item = "unknown"
    out = []
    if not raw or not str(raw).strip():
        return [default_item] * n_expected
    text = raw.strip()
    # Strip markdown code block if present
    if "```" in text:
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "").strip()
    try:
        m = re.search(r"\[[\s\S]*?\]", text)
        if m:
            out = json.loads(m.group())
        else:
            out = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    if not isinstance(out, list):
        out = []
    # Normalize to length n_expected
    while len(out) < n_expected:
        out.append(default_item)
    return [str(x).strip() if x is not None else default_item for x in out[:n_expected]]


def normalize_yes_no(s):
    """Normalize to 'yes' or 'no'."""
    if not s:
        return "no"
    t = str(s).lower().strip()
    if "yes" in t and "no" not in t:
        return "yes"
    if "no" in t:
        return "no"
    return "no"


def parse_exist_and_cooccur(raw, n_candidates):
    """Parse 2nd LLM output: (exists yes/no array, co_occur_no_indices array)."""
    answers = ["no"] * n_candidates
    co_occur_no_indices = []
    if not raw or not str(raw).strip():
        return answers, co_occur_no_indices
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "").strip()
    arrays = []
    for m in re.finditer(r"\[[\s\S]*?\]", text):
        try:
            arr = json.loads(m.group())
            arrays.append(arr)
        except (json.JSONDecodeError, TypeError):
            continue
    for arr in arrays:
        if not arr:
            continue
        first = arr[0]
        if isinstance(first, str) and first.lower() in ("yes", "no"):
            out = [normalize_yes_no(x) for x in arr[:n_candidates]]
            while len(out) < n_candidates:
                out.append("no")
            answers = out[:n_candidates]
        elif isinstance(first, (int, float)) or (isinstance(first, str) and str(first).strip().lstrip("-").isdigit()):
            for x in arr:
                try:
                    if isinstance(x, int):
                        co_occur_no_indices.append(x)
                    elif isinstance(x, float) and x == int(x):
                        co_occur_no_indices.append(int(x))
                    elif isinstance(x, str) and str(x).strip().lstrip("-").isdigit():
                        co_occur_no_indices.append(int(x))
                except (ValueError, TypeError):
                    pass
    return answers, co_occur_no_indices


def main():
    parser = argparse.ArgumentParser(
        description="Convert conv_script to JSON with dsg_qa: 2 LLM calls per conversation (objects from context, then exist-match)."
    )
    parser.add_argument("--conv_script", type=str, required=True, help="Input conversation JSON.")
    parser.add_argument("--outfile", type=str, required=True, help="Output JSON with dsg_qa filled.")
    parser.add_argument("--pope_model_name", type=str, default=REMOTE_MODEL, help="LLM for both calls.")
    parser.add_argument("--context_key", type=str, default="context", help="Key for context dict (background, goal).")
    parser.add_argument("--batch_size", type=int, default=5, help="Samples per batch.")
    parser.add_argument("--sample_num", type=int, default=None, help="Max samples to process.")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index.")
    parser.add_argument("--resume", action="store_true", help="Skip samples that already have dsg_qa.")
    parser.add_argument("--local", action="store_true", help="Use local vLLM at localhost:8004.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    conv_path = os.path.abspath(args.conv_script)
    out_path = os.path.abspath(args.outfile)
    if conv_path == out_path:
        raise SystemExit("Error: --conv_script and --outfile must be different files.")

    global client
    if args.local:
        client = OpenAI(api_key=LOCAL_API_KEY, base_url=LOCAL_BASE_URL, timeout=API_TIMEOUT_SEC)
        if not args.pope_model_name or args.pope_model_name == REMOTE_MODEL:
            args.pope_model_name = LOCAL_MODEL
        print(f"Using local vLLM at {LOCAL_BASE_URL} with model {args.pope_model_name}")
    else:
        if not REMOTE_BASE_URL:
            raise SystemExit("Set REMOTE_API_URL in .env or use --local.")
        client = OpenAI(api_key=REMOTE_API_KEY, base_url=REMOTE_BASE_URL, timeout=API_TIMEOUT_SEC)

    get_response = partial(get_llm_response, model=args.pope_model_name)

    samples = json.load(open(args.conv_script, "r"))

    def _is_sample_complete(sample):
        conv = sample.get("conversations", [])
        n_rounds = len(conv)
        if n_rounds == 0:
            return False
        count = 0
        for turn in conv:
            for qa in turn.get("dsg_qa", []):
                if qa.get("question") and "answer" in qa:
                    count += 1
        return count >= n_rounds

    if args.resume and os.path.isfile(args.outfile):
        try:
            out_data = json.load(open(args.outfile, "r"))
            if len(out_data) != len(samples):
                print("Resume: outfile length != input; ignoring resume.")
            else:
                end = (args.start_idx + args.sample_num) if args.sample_num else len(samples)
                end = min(end, len(samples))
                for i in range(args.start_idx, end):
                    if _is_sample_complete(out_data[i]):
                        samples[i] = out_data[i]
                    else:
                        args.start_idx = i
                        break
                else:
                    args.start_idx = len(samples)
        except Exception as e:
            print(f"Resume load failed: {e}")

    end_idx = (args.start_idx + args.sample_num) if args.sample_num else len(samples)
    samples_to_process = samples[args.start_idx:end_idx]
    if not samples_to_process:
        print("Nothing to process. Exiting.")
        return

    print(f"Processing {len(samples_to_process)} samples (start_idx={args.start_idx}), 2 LLM calls; pool size={N_CANDIDATES}, output questions= n_rounds per sample (1/2 yes, 1/4 no_rand, 1/4 no_cooccur).")

    for batch_start in tqdm.tqdm(range(0, len(samples_to_process), args.batch_size), desc="Batches"):
        batch_samples = samples_to_process[batch_start : batch_start + args.batch_size]

        for sample in batch_samples:
            conversations = sample.get("conversations", [])
            n_rounds = len(conversations)
            if n_rounds == 0:
                if args.verbose:
                    print("  Skip: no conversations")
                continue

            ctx = sample.get(args.context_key) or {}
            background = ctx.get("background") or "No background given."
            goal = ctx.get("goal") or "No goal given."

            # 1) LLM call: get pool of N_CANDIDATES candidate objects from context+goal
            prompt1 = OBJECTS_FROM_CONTEXT_PROMPT.format(
                n=N_CANDIDATES,
                background=background,
                goal=goal,
            )
            try:
                raw1 = get_response(prompt1)
            except Exception as e:
                if args.verbose:
                    print(f"  LLM call 1 failed: {e}")
                raw1 = ""
            candidates = parse_json_array(raw1, N_CANDIDATES, default_item="object")

            # 2) Ground truth: existing objects from scene graph
            _names_set, existing_display = extract_object_list_from_sg(sample)
            existing_list_str = ", ".join(existing_display) if existing_display else "(no objects in scene graph)"

            # 3) LLM call: exists yes/no + which non-existing co-occur with existing list
            candidates_indexed = ", ".join(f"{i}: {c}" for i, c in enumerate(candidates))
            prompt2 = EXIST_AND_COOCCUR_PROMPT.format(
                existing_list=existing_list_str,
                n=len(candidates),
                candidates_list=candidates_indexed,
            )
            try:
                raw2 = get_response(prompt2)
            except Exception as e:
                if args.verbose:
                    print(f"  LLM call 2 failed: {e}")
                raw2 = ""
            answers, co_occur_no_indices = parse_exist_and_cooccur(raw2, len(candidates))

            # 4) Split: yes_list, no_list, no_cooccur_list (no's whose index in co_occur_no_indices)
            yes_list = [(candidates[i], "yes") for i in range(len(candidates)) if i < len(answers) and answers[i] == "yes"]
            no_list = [(candidates[i], "no") for i in range(len(candidates)) if i < len(answers) and answers[i] == "no"]
            co_occur_no_set = set(co_occur_no_indices)
            no_cooccur_list = [(candidates[i], "no") for i in range(len(candidates)) if i < len(answers) and answers[i] == "no" and i in co_occur_no_set]
            no_other_list = [(candidates[i], "no") for i in range(len(candidates)) if i < len(answers) and answers[i] == "no" and i not in co_occur_no_set]

            # 5) Composition: n_rounds total = 1/2 yes, 1/4 no_rand, 1/4 no_cooccur
            n_yes = n_rounds // 2
            n_no_rand = n_rounds // 4
            n_no_cooccur = n_rounds - n_yes - n_no_rand
            if n_no_cooccur < 0:
                n_no_cooccur = 0
                n_no_rand = n_rounds - n_yes
            random.shuffle(yes_list)
            random.shuffle(no_other_list)
            random.shuffle(no_cooccur_list)
            sampled_yes = yes_list[:n_yes]
            sampled_no_rand = no_other_list[:n_no_rand]
            sampled_no_cooccur = no_cooccur_list[:n_no_cooccur]
            combined = sampled_yes + sampled_no_rand + sampled_no_cooccur
            # Pad if short: prefer more no_rand then no_cooccur then yes
            while len(combined) < n_rounds:
                need = n_rounds - len(combined)
                if len(no_other_list) > len(sampled_no_rand):
                    take = min(need, len(no_other_list) - len(sampled_no_rand))
                    combined.extend(no_other_list[len(sampled_no_rand) : len(sampled_no_rand) + take])
                    sampled_no_rand = no_other_list[: len(sampled_no_rand) + take]
                elif len(no_cooccur_list) > len(sampled_no_cooccur):
                    take = min(need, len(no_cooccur_list) - len(sampled_no_cooccur))
                    combined.extend(no_cooccur_list[len(sampled_no_cooccur) : len(sampled_no_cooccur) + take])
                    sampled_no_cooccur = no_cooccur_list[: len(sampled_no_cooccur) + take]
                elif len(yes_list) > len(sampled_yes):
                    take = min(need, len(yes_list) - len(sampled_yes))
                    combined.extend(yes_list[len(sampled_yes) : len(sampled_yes) + take])
                else:
                    break
            combined = combined[:n_rounds]
            random.shuffle(combined)

            # 6) Build dsg_qa: one item per turn (n_rounds total), original-format keys
            for turn_idx in range(n_rounds):
                if turn_idx < len(combined):
                    obj, ans = combined[turn_idx]
                    conversations[turn_idx]["dsg_qa"] = [
                        {"index": turn_idx, "question": f"is there a {obj} in the image?", "answer": ans}
                    ]
                else:
                    conversations[turn_idx]["dsg_qa"] = [
                        {"index": turn_idx, "question": "is there an object in the image?", "answer": "no"}
                    ]

        with open(args.outfile, "w") as f:
            json.dump(samples, f, indent=2)

    print(f"Saved to {args.outfile}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Partial results may be in outfile.")
        sys.exit(130)
