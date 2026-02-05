import torch
import os
import argparse
import json
import tqdm
from PIL import Image
import copy
import requests
from io import BytesIO


import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

def eval_model(processor, model, image_file, query, conversation_history=None):
    # Track whether conversation_history was originally None to determine return type
    was_none = conversation_history is None
    
    if conversation_history is not None and len(conversation_history) > 0:
        # Use existing conversation history
        # In later rounds, only send text query (no image)
        # debug injection
        if len(conversation_history) % 6 == 0:
            conversation_history.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Please summarize our conversation so far. Do you see a image and when did you see that?"}
                    ]
                }
            )
        else:
            conversation_history = copy.deepcopy(conversation_history)
            conversation_history.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query}
                    ]
                }
            )
    else:
        # Initialize conversation_history as a new list if it's None or empty
        # First round: include image
        conversation_history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image"},
                ],
            },
        ]
    
    prompt = processor.apply_chat_template(conversation_history, add_generation_prompt=True)
    
    # Only process image when conversation history contains image token
    # First round: was_none=True, so conversation_history has image
    # Debug injection rounds: conversation_history has image token added
    # Normal later rounds: conversation_history only has text, no image
    has_image_in_history = any(
        isinstance(msg.get("content"), list) and 
        any(item.get("type") == "image" for item in msg.get("content", []))
        for msg in conversation_history
    )
    
    if has_image_in_history:
        # Include image when conversation history contains image token
        inputs = processor(images=image_file, text=prompt, return_tensors='pt').to(0, torch.float16)
    else:
        # Later rounds: only text, no image processing
        # LLaVA processor may require images parameter - try without it first
        # If this fails, the conversation wrapper should store the image from first round
        try:
            inputs = processor(text=prompt, return_tensors='pt').to(0, torch.float16)
        except TypeError:
            # If processor requires images parameter, we may need to pass None or empty
            # This is a fallback - ideally the processor should handle text-only prompts
            inputs = processor(images=None, text=prompt, return_tensors='pt').to(0, torch.float16)

    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    output_text = processor.decode(output[0][2:], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

    # Return tuple only if conversation_history was originally provided (not None)
    if not was_none:
        conversation_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": output_text}]
        })
        return output_text, conversation_history
    else:
        return output_text





