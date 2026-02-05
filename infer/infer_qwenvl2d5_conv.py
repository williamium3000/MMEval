import torch
import os
import argparse
import json
import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

def eval_model(processor, model, image_file, query, conversation_history=None, debug=False, stored_image=None):
    """
    Evaluate model with support for multi-turn conversations.
    
    Args:
        processor: The Qwen2.5-VL processor
        model: The Qwen2.5-VL model
        image_file: Path to the image file
        query: The user query/question
        conversation_history: Optional list of previous messages in the conversation.
                            If None, starts a new conversation.
                            Format: [{"role": "user", "content": [...]}, {"role": "assistant", "content": "..."}, ...]
        debug: If True, print debug information about conversation state
        stored_image: The actual image object to use (passed from wrapper to ensure consistency)
    
    Returns:
        tuple: (output_text, updated_conversation_history, stored_image)
    """
    # Initialize conversation history (accumulate both user and assistant messages)
    if conversation_history is None:
        conversation_history = []
    
    if debug:
        print(f"\n[DEBUG] image_file type: {type(image_file)}, value: {image_file}")
        print(f"[DEBUG] stored_image type: {type(stored_image)}, value: {stored_image}")
    
    # Include image in every user message (required for process_vision_info to work)
    # This matches the reference implementation pattern
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_file,
            },
            {"type": "text", "text": query},
        ],
    }
    
    # Store image on first call for consistency
    if stored_image is None:
        stored_image = image_file
    
    # Add current user message to conversation history
    conversation_history.append(user_message)

    # Use full conversation history for chat template (includes all previous turns)
    text = processor.apply_chat_template(
        conversation_history, tokenize=False, add_generation_prompt=True
    )
    # Extract all images/videos from conversation history
    image_inputs, video_inputs = process_vision_info(conversation_history)
    
    if debug:
        print(f"[DEBUG] Extracted image_inputs: {type(image_inputs)}, length: {len(image_inputs) if image_inputs else 0}")
        if image_inputs:
            print(f"[DEBUG] First image: {type(image_inputs[0]) if len(image_inputs) > 0 else 'None'}")
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Add assistant response to conversation history
    conversation_history.append({
        "role": "assistant",
        "content": output_text
    })
    
    # Debug mode: on 3rd round, inject a question asking to summarize conversation history
    if debug and len(conversation_history) % 6 == 0:  # After 3rd Q&A pair
        print("\n[DEBUG] Injecting conversation summary question...")
        
        # Add summary question with image
        summary_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": stored_image,
                },
                {"type": "text", "text": "Please summarize our conversation so far. Do you see a image and when did you see that?"},
            ],
        }
        conversation_history.append(summary_message)
        
        # Get response to summary question
        summary_text = processor.apply_chat_template(
            conversation_history, tokenize=False, add_generation_prompt=True
        )
        summary_image_inputs, summary_video_inputs = process_vision_info(conversation_history)
        summary_inputs = processor(
            text=[summary_text],
            images=summary_image_inputs,
            videos=summary_video_inputs,
            padding=True,
            return_tensors="pt",
        )
        summary_inputs = summary_inputs.to("cuda")
        
        summary_generated_ids = model.generate(**summary_inputs, max_new_tokens=256)
        summary_generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(summary_inputs.input_ids, summary_generated_ids)
        ]
        summary_output = processor.batch_decode(
            summary_generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"[DEBUG] Summary response: {summary_output}\n")
        
        # Add summary response to history
        conversation_history.append({
            "role": "assistant",
            "content": summary_output
        })
    
    return output_text, conversation_history, stored_image


class ConversationWrapper:
    """
    Wrapper class to manage conversation history for multi-turn dialogues.
    This maintains separate conversation histories per image/context.
    """
    def __init__(self, model, processor, eval_func, debug=True):
        self.model = model
        self.processor = processor
        self.eval_func = eval_func
        self.debug = debug
        # Dictionary to store conversation histories keyed by PIL Image object id
        self.conversation_histories = {}
        # Dictionary to store the actual image objects for each conversation
        self.stored_images = {}
    
    def __call__(self, image_file, query):
        """
        Call the model with conversation history management.
        Maintains separate conversation histories per image.
        
        Args:
            image_file: PIL Image object (used as key for conversation history via object id)
            query: The user query/question
        
        Returns:
            str: The model's response
        """
        # Use object id as key for PIL Image objects
        image_key = id(image_file)
        
        # Get or initialize conversation history for this image
        history = self.conversation_histories.get(image_key, None)
        stored_image = self.stored_images.get(image_key, None)
        
        # Call the eval function with conversation history
        output_text, updated_history, updated_stored_image = self.eval_func(
            processor=self.processor,
            model=self.model,
            image_file=image_file,
            query=query,
            conversation_history=history,
            debug=self.debug,
            stored_image=stored_image
        )
        
        # Update the stored conversation history and image
        self.conversation_histories[image_key] = updated_history
        self.stored_images[image_key] = updated_stored_image
        
        return output_text
    
    def reset_conversation(self, image_file=None):
        """
        Reset conversation history for a specific image or all images.
        
        Args:
            image_file: If provided, reset only this image's history.
                       If None, reset all conversation histories.
        """
        if image_file is None:
            self.conversation_histories = {}
            self.stored_images = {}
        else:
            image_key = id(image_file)
            self.conversation_histories.pop(image_key, None)
            self.stored_images.pop(image_key, None)




