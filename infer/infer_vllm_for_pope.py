#!/usr/bin/env python3
"""
Infer VLM responses for POPE questions using vLLM API endpoint.
Reads extracted questions JSON and calls vLLM endpoint for each question.
Outputs _pope_output.json with 'output' field added to each question.
"""

import json
import os
import argparse
import requests
from typing import List, Dict
from tqdm import tqdm
import base64
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO


def prepare_image_content(image_path_or_url: str) -> dict:
    """
    Prepare image content for API call.
    Downloads images from URLs and encodes as base64, or reads local files.
    Ensures proper multimodal input format for vLLM API.
    
    Args:
        image_path_or_url: URL or local file path to image
    
    Returns:
        Dictionary with image content format (base64 encoded)
    """
    image_bytes = None
    mime_type = 'image/jpeg'
    
    # Check if it's a URL
    parsed = urlparse(image_path_or_url)
    if parsed.scheme in ('http', 'https'):
        # Download image from URL
        try:
            response = requests.get(image_path_or_url, timeout=30)
            response.raise_for_status()
            image_bytes = response.content
            
            # Try to determine MIME type from Content-Type header
            content_type = response.headers.get('Content-Type', '')
            if 'image/png' in content_type:
                mime_type = 'image/png'
            elif 'image/jpeg' in content_type or 'image/jpg' in content_type:
                mime_type = 'image/jpeg'
            elif 'image/gif' in content_type:
                mime_type = 'image/gif'
            elif 'image/webp' in content_type:
                mime_type = 'image/webp'
            else:
                # Try to detect from image data
                try:
                    img = Image.open(BytesIO(image_bytes))
                    mime_type = f'image/{img.format.lower()}' if img.format else 'image/jpeg'
                except:
                    mime_type = 'image/jpeg'
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image from URL {image_path_or_url}: {e}")
            raise
    else:
        # It's a local file path
        if os.path.exists(image_path_or_url):
            # Read local file
            with open(image_path_or_url, 'rb') as f:
                image_bytes = f.read()
            
            # Determine MIME type from extension
            ext = os.path.splitext(image_path_or_url)[1].lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }.get(ext, 'image/jpeg')
        else:
            raise FileNotFoundError(f"Image not found: {image_path_or_url}")
    
    # Encode image as base64
    if image_bytes is None:
        raise ValueError("Failed to load image data")
    
    image_data = base64.b64encode(image_bytes).decode('utf-8')
    
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{image_data}"
        }
    }


def call_vllm_api(api_url: str, image_path_or_url: str, question: str, model_name: str = None, api_key: str = None) -> str:
    """
    Call vLLM API endpoint with image and question.
    
    IMPORTANT: The image is included in EVERY API call. Each question is processed
    independently with the image attached, ensuring the model sees the image for
    every single question.
    
    Args:
        api_url: vLLM API endpoint URL (e.g., http://localhost:8000/v1/chat/completions)
        image_path_or_url: URL or local file path to the image
        question: Question text
        model_name: Optional model name parameter
        api_key: Optional API key; if vLLM was started with --api-key, pass the same key here.
                 Omit for local vLLM with no auth (default).
    
    Returns:
        Response text from the model
    """
    # Prepare image content (handles both URLs and local paths)
    # The image is included in every single API call
    image_content = prepare_image_content(image_path_or_url)
    
    # Prepare messages in OpenAI-compatible format
    # Each message includes both the image and the question text
    messages = [
        {
            "role": "user",
            "content": [
                image_content,  # Image is always included
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ]
    
    payload = {
        "model": model_name or "default",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.0
    }
    
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        response = requests.post(api_url, json=payload, headers=headers or None, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        # Extract text from response
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            print(f"Warning: Unexpected response format: {result}")
            return ""
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return ""


def process_questions(input_file: str, output_file: str, api_url: str, model_name: str = None, api_key: str = None, batch_size: int = 1, quiet: bool = False):
    """
    Process all questions in the input file and add 'output' field.
    
    Args:
        input_file: Path to extracted questions JSON
        output_file: Path to output JSON with 'output' field
        api_url: vLLM API endpoint URL
        model_name: Optional model name
        api_key: Optional API key; use if vLLM server was started with --api-key
        batch_size: Number of requests to process (currently 1, can be extended for batching)
        quiet: If True, suppress progress bar (useful for parallel execution)
    """
    file_basename = os.path.basename(input_file)
    if not quiet:
        print(f"Loading questions from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    if not quiet:
        print(f"Found {len(questions)} questions")
        print(f"API URL: {api_url}")
        print(f"Model: {model_name or 'default'}")
    
    results = []
    
    # Process each question
    # IMPORTANT: Each question is processed independently with the image included.
    # The image is sent with EVERY question (not just the first one).
    # Include filename in progress bar description for parallel execution visibility
    progress_desc = f"Processing {file_basename[:30]}"
    iterator = tqdm(questions, desc=progress_desc, disable=quiet) if not quiet else questions
    for item in iterator:
        question = item.get("question", "")
        image_url = item.get("image", "")
        
        if not question or not image_url:
            print(f"Warning: Skipping item with missing question or image: {item.get('index')}")
            item["output"] = ""
            results.append(item)
            continue
        
        # Call vLLM API - image is included in every call
        # Each question gets a fresh API call with the image attached
        output = call_vllm_api(api_url, image_url, question, model_name, api_key=api_key)
        item["output"] = output
        results.append(item)
    
    # Save results
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Processed {len(results)} questions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Infer VLM responses for POPE questions using vLLM API')
    parser.add_argument('input_file', type=str, help='Input extracted questions JSON file')
    parser.add_argument('--api-url', type=str, required=True, 
                       help='vLLM API endpoint URL (e.g., http://localhost:8000/v1/chat/completions)')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Model name to use (optional, defaults to API default)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API key if vLLM was started with --api-key. Default: no auth (for local vLLM). Can also set VLLM_API_KEY env.')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (default: input_file with _pope_output suffix)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress bar (useful for parallel execution)')
    
    args = parser.parse_args()
    api_key = args.api_key or os.environ.get('VLLM_API_KEY', '').strip() or None
    
    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found")
        exit(1)
    
    if args.output is None:
        base_name = args.input_file.replace('_extracted.json', '').replace('.json', '')
        output_file = f"{base_name}_pope_output.json"
    else:
        output_file = args.output
    
    process_questions(args.input_file, output_file, args.api_url, args.model_name, api_key=api_key, quiet=args.quiet)
