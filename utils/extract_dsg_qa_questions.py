#!/usr/bin/env python3
"""
Extract all dsg_qa questions from _pope_converted.json files and convert to a flat format.
Output format:
[
    {
        "index": 1,
        "question": "Is there a parking meter?",
        "answer": "Yes",
        "has_answer": true,
        "evidence": "[Turn 1] ANSWER: ...",
        "tuple": "entity - whole",
        "image_id": 1,
        "round_id": 1,
        "image": "https://..."
    },
    ...
]
"""

import json
import os
import argparse
import sys
import traceback
import re


def load_json_robust(input_file):
    """
    Load JSON file with robust error handling for corrupted files.
    Attempts multiple strategies to recover data.
    """
    error_msg = None
    error_pos = None
    
    # Strategy 1: Try normal JSON loading
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data, None
    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error at line {e.lineno}, column {e.colno}: {e.msg}"
        error_pos = getattr(e, 'pos', None)
        print(f"Warning: {error_msg}")
        print(f"Attempting recovery strategies...")
    except Exception as e:
        error_msg = f"Error loading JSON: {str(e)}"
        print(f"Warning: {error_msg}")
        print(f"Attempting recovery strategies...")
    
    # Strategy 2: Try reading as JSONL (one JSON object per line)
    try:
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # Only add if it's a dict (not a string or other type)
                    if isinstance(item, dict):
                        data.append(item)
                except json.JSONDecodeError:
                    # Skip corrupted lines silently
                    continue
        if data:
            print(f"  Recovered {len(data)} items from JSONL format")
            return data, None
    except Exception as e:
        print(f"  JSONL recovery failed: {e}")
    
    # Strategy 3: Try reading up to the error position and find last complete item
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if error_pos is None:
            error_pos = len(content)
        
        # Find the last complete "image_id" entry
        # Look for pattern: "image_id": <number> followed by closing brace
        # Find all occurrences of "image_id": followed by a number
        # Then check if there's a complete object structure after it
        pattern = r'"image_id"\s*:\s*(\d+)'
        matches = list(re.finditer(pattern, content[:error_pos]))
        
        if matches:
            # Start from the last match and work backwards
            for match in reversed(matches):
                match_end = match.end()
                # Look for the closing brace of this object
                # Find the next '}' that closes this object (not nested)
                brace_pos = match_end
                depth = 0
                found_closing = False
                last_valid_pos = None
                
                # Search forward from the image_id to find the object's closing brace
                i = match_end
                while i < min(error_pos, match_end + 5000):  # Limit search range
                    if content[i] == '{':
                        depth += 1
                    elif content[i] == '}':
                        depth -= 1
                        if depth == 0:
                            # Found the closing brace of the object containing this image_id
                            last_valid_pos = i + 1
                            found_closing = True
                            break
                    i += 1
                
                if found_closing and last_valid_pos:
                    # Check if there's a comma after this object (meaning it's not the last item)
                    # Or if it's the last item, we need to add the closing bracket
                    partial_content = content[:last_valid_pos]
                    
                    # Remove any trailing comma and whitespace
                    partial_content = partial_content.rstrip().rstrip(',')
                    
                    # Try to parse as complete JSON array
                    try_close = partial_content + '\n]'
                    try:
                        data = json.loads(try_close)
                        print(f"  Recovered {len(data)} items by finding last complete image_id at position {last_valid_pos}")
                        return data, f"File truncated at position {last_valid_pos} (last complete image_id: {match.group(1)})"
                    except json.JSONDecodeError as e3:
                        # If that didn't work, try finding the last complete object differently
                        # Look for the pattern: }, followed by whitespace and {
                        # This indicates a complete item followed by start of next
                        last_complete_pattern = r'\}\s*,\s*\{'
                        complete_matches = list(re.finditer(last_complete_pattern, content[:error_pos]))
                        if complete_matches:
                            last_complete = complete_matches[-1]
                            # The position after the }, is where we want to cut
                            cut_pos = last_complete.end() - 1  # Position of the } before the comma
                            try_close2 = content[:cut_pos + 1] + '\n]'
                            try:
                                data = json.loads(try_close2)
                                print(f"  Recovered {len(data)} items by finding last complete item separator")
                                return data, f"File truncated at position {cut_pos + 1}"
                            except:
                                pass
                        continue
        
        # Fallback: Try simple truncation at last closing bracket
        search_start = max(0, error_pos - 10000)
        last_bracket = content.rfind(']', search_start, error_pos)
        if last_bracket > 0:
            partial_content = content[:last_bracket + 1]
            try:
                data = json.loads(partial_content)
                print(f"  Recovered {len(data)} items by truncating at last bracket")
                return data, f"File truncated at position {last_bracket}"
            except:
                pass
                
    except Exception as e2:
        print(f"  Truncation recovery failed: {e2}")
    
    # All strategies failed
    return None, f"Failed to load JSON file. Original error: {error_msg or 'Unknown error'}"


def extract_dsg_qa_questions(input_file, output_file):
    """Extract all dsg_qa questions from a pope JSON file and convert to flat format"""
    print(f"\nProcessing: {os.path.basename(input_file)}")
    print("Loading JSON file...")
    
    data, error = load_json_robust(input_file)
    if data is None:
        print(f"Error: {error}")
        print(f"Skipping file: {input_file}")
        return 0
    
    if error:
        print(f"Warning: {error}")
    
    print(f"Processing {len(data)} images...")
    result = []
    index_counter = 1
    
    # Iterate through each image
    for image_data in data:
        # Skip if not a dict (shouldn't happen, but be safe)
        if not isinstance(image_data, dict):
            continue
            
        image_id = image_data.get('image_id')
        image_url = image_data.get('url')
        
        # Iterate through conversations
        conversations = image_data.get('conversations', [])
        for conversation in conversations:
            round_id = conversation.get('round_id')
            dsg_qa = conversation.get('dsg_qa', [])
            
            # Iterate through each dsg_qa item
            for qa_item in dsg_qa:
                # Create a new dictionary with required fields
                new_item = {
                    "index": qa_item.get("index") or qa_item.get("id") or index_counter,
                    "question": qa_item.get("question", ""),
                    "answer": qa_item.get("answer", "Unknown"),
                    "has_answer": qa_item.get("has_answer", False),
                    "evidence": qa_item.get("evidence", ""),
                    "tuple": qa_item.get("tuple", ""),
                    "image_id": image_id,
                    "round_id": round_id,
                    "image": image_url
                }
                result.append(new_item)
                index_counter += 1
    
    print(f"Extracted {len(result)} questions")
    print(f"Saving to {os.path.basename(output_file)}...")
    
    # Save the result
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("Done!")
    if result:
        print(f"Sample output (first item):")
        print(json.dumps(result[0], indent=2, ensure_ascii=False))
    
    return len(result)


def run_extract_only(dir_path):
    """Find all *_pope_converted.json under dir_path and run extraction for each."""
    if not os.path.isdir(dir_path):
        print(f"Error: {dir_path} is not a directory")
        sys.exit(1)
    converted = []
    for root, _dirs, files in os.walk(dir_path):
        for f in files:
            if f.endswith('_pope_converted.json'):
                converted.append(os.path.join(root, f))
    if not converted:
        print(f"No *_pope_converted.json files found under {dir_path}")
        return
    print(f"Found {len(converted)} _pope_converted file(s), extracting...")
    for path in sorted(converted):
        base_name = path.replace('_pope_converted.json', '').replace('.json', '')
        output_file = f"{base_name}_extracted.json"
        extract_dsg_qa_questions(path, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract dsg_qa questions from _pope_converted.json files')
    parser.add_argument('input_file', type=str, nargs='?', default=None, help='Input _pope_converted.json file')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file (default: input_file with _extracted suffix)')
    parser.add_argument('--extract', type=str, nargs='?', const='work_dirs/vg', default=None,
                        metavar='DIR', help='Extract-only mode: find all *_pope_converted.json under DIR (default: work_dirs/vg) and extract each')
    
    args = parser.parse_args()
    
    if args.extract is not None:
        run_extract_only(args.extract)
        sys.exit(0)
    
    if args.input_file is None:
        parser.error("input_file is required unless --extract is used")
    
    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found")
        sys.exit(1)
    
    if args.output is None:
        base_name = args.input_file.replace('_pope_converted.json', '').replace('.json', '')
        output_file = f"{base_name}_extracted.json"
    else:
        output_file = args.output
    
    extract_dsg_qa_questions(args.input_file, output_file)
