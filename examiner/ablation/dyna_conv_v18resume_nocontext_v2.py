# 1) change switch/follow-up back to general, and keep interrogate requirement only in prompt
# 2) parse string formats of relevant object nodes

from utils.utils import load_data
from utils.vg import format_case_vg
from utils.coco import format_case_coco
from utils.llm import LLMChat, parse_json
from examiner import prompt as PROMPT
from infer.loader import load_model
import os
import argparse
import json
import tqdm
import copy
from utils.sg import SceneGraphData
import random
import traceback
import re

# Context generation removed - using general investigation goal


SWITCH_PROMPT_EARLY = \
"""Based on the given conversation history, please decide which type of question to ask next. Please choose one of the following types of questions:

1. **Regular questions** – directly related to investigating instances, attributes, or relations in the image.
    *Example*: "What color is the object on the left?", "Where is the largest object located?", "What is the relationship between the two objects?"

2. **Follow-up questions** – follow up, confirm or interrogate the model's last response to test the confidence of the model's answer or to ask for further challenging details when available.
   *Example*: If the model correctly identifies there's no fork on the table, you may ask: *"Are you sure there isn't a fork on the table?"*

3. **Adversarial questions** – inquire about plausible but absent objects that commonly co-occur with visible ones in the image.
    *Example*: If the image shows a cake without utensils, you may ask: *"Can I use the knife on the table to cut the cake?"* or *"Is there a fork?"*

4. **Unanswerable questions** – ask about question that cannot be answered.
   *Example*: if the image depicts a cake on the table without any utensil or people eating the cake, you can ask "What utensil is the man using to cut the cake?". This is an unanswerable question because you cannot answer with an utensil but rather you should answer "There are not any man in the image eating the cake".

Given the previously asked question types: {}, try to ask diverse types of questions.

YOU CAN ONLY SELECT ONE OF THE ABOVE FOUR TYPES OF NEXT-STEPS !!
Respond format: think thoroughly to reason or perform chain-of-thought to select the next question type. 
However, you MUST eventually provide a digit between 1 and 4 in below json format to instruct the next round:
```json
{{"type": type_id}}
```
"""

SWITCH_PROMPT_LATE = \
"""Based on the given conversation history, please decide which type of question to ask next. Please choose one of the following types of questions:

1. **Regular questions** – directly related to investigating instances, attributes, or relations in the image.
    *Example*: "What color is the object on the left?", "Where is the largest object located?", "What is the relationship between the two objects?"

2. **Follow-up questions** – follow up, confirm or interrogate the model's last response to test the confidence of the model's answer or to ask for further challenging details when available.
   *Example*: If the model correctly identifies there's no fork on the table, you may ask: *"Are you sure there isn't a fork on the table?"*

3. **Adversarial questions** – inquire about plausible but absent objects that commonly co-occur with visible ones in the image.
    *Example*: If the image shows a cake without utensils, you may ask: *"Can I use the knife on the table to cut the cake?"* or *"Is there a fork?"*

4. **Unanswerable questions** – ask about question that cannot be answered.
   *Example*: if the image depicts a cake on the table without any utensil or people eating the cake, you can ask "What utensil is the man using to cut the cake?". This is an unanswerable question because you cannot answer with an utensil but rather you should answer "There are not any man in the image eating the cake".

5. **End the conversation** – to end the conversation.
   **IMPORTANT**: ONLY choose this option if you believe all instances, attributes, and relations in the image have been thoroughly explored by the 4 dimensions of questions above.

Given the previously asked question types: {}, try to ask diverse types of questions.

YOU CAN ONLY SELECT ONE OF THE ABOVE FIVE TYPES OF NEXT-STEPS !!
Respond format: think thoroughly to reason or perform chain-of-thought to select the next question type. 
However, you MUST eventually provide a digit between 1 and 5 (with 5 representing "End the conversation") in below json format to instruct the next round:
```json
{{"type": type_id}}
```
"""

CONV_SYSTEM_PROMPT = \
"""Your task is to **simulate a multi-round conversation with a vision-language model (VLM) about a given image**, in order to test whether the model hallucinates (i.e., produces responses inconsistent with the image) or remains faithful to it.
**Ultimate Goal:** Fully investigate all instances, attributes, and relations in the image by raising challenging visual questions and probing as many hallucinations from the responding model as possible.

You will have multiple rounds of conversations with the model. In each round, you will be provided with:

* **An image** (represented as a list of objects, their attributes, and relationships).
* **Bounding-box coordinates** for each object, given as `(x1, y1, x2, y2)` in normalized values between 0 and 1, corresponding to top-left and bottom-right corners of each object.

Your role is to **carry out a natural, open-ended, human-like conversation** with the model, systematically asking questions to explore all instances, attributes, and relations in the image.

Requirements

1. **Multi-turn conversation**:

    * Ask one question per turn, then wait for the model’s response.
    * Incorporate the image and dialogue history when forming your question.

2. **Natural, human-like tone**:

    * Speak as if conversing casually with another person.
    * Avoid mechanical or scripted phrasing.

3. **No disclosure of metadata**:

    * Do not reveal or reference bounding boxes, object lists, captions, or the source of your information.
    * Ask questions as if you are looking at the image with NO access to the bounding boxes, object lists, or captions.

4. **Handling mistakes**:

    * Do not correct the model if they make mistakes. 
    * However, you may ask follow-ups to probe or interrogate its response.

5. **Question style**:

    * Prefer open-ended questions over yes/no questions.
    * Ask diverse, non-redundant questions.
    * Only ask questions with **definite answers** (either clearly present in the image or confidently absent).

6. **Conversation ending**:

    * If the dialogue has naturally run its course and all instances, attributes, and relations have been explored, output **"END"** (and nothing else).

Image information:
{}
"""

REGULAR_CONV_PROMPT = \
"""You are tasked with generating a *regular question* grounded in the provided image.

* **Target node:** `{}`

**Instructions:**

* Ask a natural, conversational question about the specified node.
* Avoid irrelevant, generic, or out-of-character questions.
* Keep the question consistent with the role of someone actively engaged in the situation.
* Generate the ground-truth answer based on the given image content along the question.
* **CRITICAL: Do NOT repeat or ask questions with the same meaning as any prior questions in the conversation history.**
* **STRICTLY AVOID asking about the same object that has already been asked about in previous questions. Try to ask about different objects to maximize diversity.**

**Previously asked questions (DO NOT repeat these or ask similar questions, and try NOT to ask about the same objects):**
{}

**Example:**
If the background is an office and the goal is “send an email to the boss,” suitable questions include:

* {{"question": "Is the monitor turned on?", "gt": "No"}}
* {{"question": "Where is the computer’s power button?", "gt": "It's on the computer under the table to the right of the image."}}
* {{"question": "I want to type the email—what should I look for?", "gt": "You should look for the keyboard and mouse on the table in front of the monitor."}}

**Output Requirement:**
Respond in the following json format:
```json
{{"question": "xxxx", "gt": "xxxx"}},
```

Respond with **the question and gt ONLY**. Do not include explanations, commentary, or any additional text.
"""

FOLLOW_UP_CONV_PROMPT = \
"""Based on the given image and the conversation history, please ask a follow-up question about the model's last turn of conversation and the corresponding ground-truth answer.
The follow-up question can be to ask for further challenging details when available or to interrogate the model's confidence in its previous correct answer.

**Interrogate Type Requirement:**
Only apply this follow-up type if the model correctly answered the previous question. We then want to test whether the confidence of the model's answer is low despite misleading subjective or peripheral evidence.
- Challenge or verify the model's previous answer
- Probe for more details
- Test consistency
- Question contradictions

**Example follow-ups:**
- After any answer: "Are you sure about [the specific detail they mentioned]?" like "Are you sure the color is [the color they mentioned]? I see it's actually [random other color]."
- Challenging with visible evidence if last answer is non-existent: "I actually see a [the questioned item] near/around [co-occurring object/resonable location], are you sure you can't see [the questioned item]?"
- Requesting clarification: "You mentioned [X], but can you describe it more precisely?"
- Testing consistency: "Earlier you said [Y], how does that relate to what you just said?"

You should ask the question as if you are having a conversation with the model. You should also follow the above requirements.
Please respond in the following format:
```json
{{"question": "xxxx", "gt": "xxxx"}},
```
Please respond with the question and the ground-truth answer ONLY. DO NOT respond with anything else.
"""


ADVERSARIAL_CONV_PROMPT1 = \
"""Based on the given image and the conversation history, please ask an adversarial question about ONE plausible scenario/object that commonly co-occur with visible feature(s) in the image but currently isn't present (the ground truth answer is always 'No').

Requirements:
1) Your generated hallucinated scenario/object should not be present in the image but should highly co-occur with image content.
2) ENCOURAGED if possible: Include an inductive clause that briefly mentions visible objects as part of the existence question (e.g., "To help cut the cake on the table, is there a knife present?").
3) **CRITICAL: Do NOT repeat or ask questions that are similar to any prior questions in the conversation history.**
4) **STRICTLY AVOID asking about the same object that has already been asked about in previous questions. Try to ask about different objects to maximize diversity.**

**Previously asked questions (DO NOT repeat these or ask similar questions, and try NOT to ask about the same objects):**
{}

Please output the hallucinated scenario/object as scene graph elements"names"/"attributes"/"relations" and the corresponding question in the form of a json dict.

Examples:
If there is a blue banana on the table while banana should usually be yellow.
```json
{{"names": "banana", "attributes": "yellow", "question": "Is there a yellow banana present on the table?"}}
```
If there is cake on the table, which is usually accompanied by a knife but currently isn't.
```json
{{"names": "knife", "question": "Is there a knife present on the table?"}}
```

Allowed keys in the JSON output are "names", "attributes", "relations", and "question".
"""

# ADVERSARIAL_CONV_PROMPT2 = \
# """Now you should ask a adversarial question based on the generated hallucinated scenario/object and the corresponding ground-truth answer.
# Your generated adversarial question should try to flow naturally with the conversation history.
# ENCOURAGED: Include an inductive clause that briefly mentions 1 or 2 of the "co-occur_with" items as part of the existence question (e.g., "Given the cake on the table, is there a knife present?").

# Please respond in the following format:
# ```json
# {{"question": "xxxx", "gt": "No", "co-occur_with": "xxxx"}},
# ```
# Where gt is always 'No' (item is not present) and co-occur_with is the comma-separated list of visible features from the previous step.
# You should ask a question as if you are having a conversation with the model. Please respond with the question and the ground-truth answer ONLY. DO NOT respond with anything else.
# """

UNANSWERABLE_CONV_PROMPT1 = \
"""Based on the given image and conversation history, generate an **unanswerable question**.

An *unanswerable question* refers to a query that cannot be answered using the provided information because it introduces a plausible but absent or incorrect object, attribute, or relation.
For example, if the image shows only a cake on the table, asking *"What utensil is the man using to cut the cake?"* is unanswerable since no man is present.

**Procedure:**

1. **Hallucinated Object:** Generate a plausible but absent or incorrect object that would typically co-occur with image content. Output in JSON format:

   ```json
   {{"names": "object_name"}}
   ```

2. **Hallucinated Relation/Attribute:** Generate a plausible relation or attribute involving the hallucinated object. Here the relation should link the hallucinated object to an actual object from the image. Output in JSON format:

   ```json
   {{"relations": "relation", "object": "real_object", "subject": "hallucinated_object"}}
   ```

3. **Unanswerable Question:** Formulate a natural question about this hallucinated relation or attribute. The question should sound plausible but must be unanswerable from the provided image.

**Example Walkthrough:**

* Image: A cake on the table, no people visible.
* Step 1:

  ```json
  {{"names": "man"}}
  ```
* Step 2:

  ```json
  {{"relations": "eating", "object": "cake", "subject": "man"}}
  ```
* Step 3: ask the unanswerable question about the relation with corresponding ground-truth answer: "eating" between the generated object "man" and the object "cake": “What utensil is the man using to cut the cake?”*

  ```json
  {{"question": "What utensil is the man using to cut the cake?", "gt": "There are not any man in the image eating the cake"}}
  ```
---

**CRITICAL - Avoid Repetition:**
**The SAME hallucinated object in previously asked unanswerable questions also counts as repetition. For example, if you previously asked about "the man's shirt color", do NOT ask about "the man's position" or any other feature of "the man". You MUST introduce a NEW hallucinated object.**
{}


**Your Task Now:**
Complete **Step 1** by generating a plausible but absent or incorrect object in JSON format.
"""

UNANSWERABLE_CONV_PROMPT2 = \
"""Now complete Step 2. Generate a plausible attribute or relation for the hallucinated object you created in Step 1. This relation must link the hallucinated object to one of the real objects present in the image."""

UNANSWERABLE_CONV_PROMPT3 = \
"""Now complete Step 3. You must ask an **UNANSWERABLE question** about the hallucinated relation/attribute you generated, and provide the corresponding ground-truth answer (gt).

CRITICAL GOAL (Separation Guarantee):
This must be an **UNANSWERABLE (presuppositional trap) question** that **DOES presuppose** the hallucinated object/relation/attribute exists in the image.
The question must sound plausible, but it cannot be answered from the provided image because the key premise is false/absent.

Hard Requirements (MUST follow):
1) **Strong presupposition is mandatory**
   - You MUST write the question as if the hallucinated object/relation is definitely there.
   - Use definite references and presuppositional phrasing, e.g.:
     "the", "that", "the one on the table", "the person", "the knife next to..."
   - The question should demand a **specific detail** that cannot be provided if the object is absent:
     color, exact location, count, identity, orientation, precise relation, etc.

2) **No hedging / no conditional phrasing**
   - DO NOT use: "is there any", "do you see", "if there is", "might", "maybe".
   - This is a trap question: it should *assume* existence.

3) **Ground-truth answer must explicitly correct the false premise**
   - The gt MUST point out the absence / invalid premise clearly, e.g.:
     "There is no X in the image."
     "I do not see X, so I cannot answer that."
     "No person is visible interacting with Y."
   - Do not invent details to satisfy the question.

**CRITICAL - Avoid Repetition:**
**Do NOT repeat or ask questions with the same meaning as any prior questions.**
**STRICTLY AVOID asking about the same object that has already been asked about in previous questions. Try to ask about different objects to maximize diversity.**

**Previously asked questions (DO NOT repeat these or ask similar questions, and try NOT to ask about the same objects):**
{}

Good Unanswerable Question Templates (Examples):
- Definite reference + attribute detail:
  * {{"question": "What color is the handle of the knife on the table?"}}
- Definite reference + relation detail:
  * {{"question": "Which side of the cake is the fork placed on—left or right?"}}
- Definite reference + human interaction detail (only if the image has no humans; this makes it unanswerable):
  * {{"question": "What utensil is the man using to cut the cake?"}}

Bad (DISALLOWED) Examples (these would overlap with ADVERSARIAL):
- "Do you see any knife we could use?" (non-presuppositional)
- "If there is a fork, where might it be?" (conditional)

Output Requirement:
Respond in the following json format:
```json
{{"question": "xxxx"}}
```

Respond with **the question ONLY**. Do not include explanations, commentary, or any additional text.
"""

Q_TYPE_MAPPING = {
    1: "regular",
    2: "follow-up",
    3: "adversarial",
    4: "unanswerable",
    5: "end",
}
class EvalSample:
    def __init__(self, case, llm_chat_conv, eval_func):
        self.case = case
        self.image_info = format_case_vg(case) if args.dataset == "vg" else format_case_coco(case)
        self.scene_graph_data = SceneGraphData.from_dict(case)

        self.llm_chat_conv = llm_chat_conv  # GPT-4o for conversation
        self.eval_func = eval_func
        self.conversations = []
        # Track previously asked questions by type to avoid repetition
        self.repeat_ref_dict = {
            "regular": [],
            "adversarial": [],
            "unanswerable": []
        }
        # Get all object nodes for investigation
        self.all_nodes = [obj for obj in self.scene_graph_data.image_info.sg.objects]

    def switch(self, conversations, swicth_history, round_num):
        conversations = copy.deepcopy(conversations)
        # Use early version (without END option) before round 6
        switch_prompt = SWITCH_PROMPT_EARLY if round_num < 6 else SWITCH_PROMPT_LATE
        conversations.append(
            {"role": "user", "content": switch_prompt.format(swicth_history)})
        type_id = self.llm_chat_conv.chat(conversations, parse_json)['type']
        # Handle case where LLM returns "END" string instead of 5
        if isinstance(type_id, str) and type_id.upper() == "END":
            type_id = 5
        return type_id
    
    def ask_regular(self, conversations, selected_nodes):
        conversations = copy.deepcopy(conversations)
        sampled_node = selected_nodes.pop(0)
        prev_questions_str = json.dumps(self.repeat_ref_dict["regular"], indent=2) if self.repeat_ref_dict["regular"] else "None"
        conversations.append({"role": "user", "content": REGULAR_CONV_PROMPT.format(sampled_node, prev_questions_str)})
        message = self.llm_chat_conv.chat(conversations, parse_json)
        # Track this question to prevent repetition
        self.repeat_ref_dict["regular"].append(message.get("question", ""))
        return message
    
    def ask_follow_up(self, conversations):
        conversations = copy.deepcopy(conversations)
        conversations.append({"role": "user", "content": FOLLOW_UP_CONV_PROMPT})
        message = self.llm_chat_conv.chat(conversations, parse_json)
        return message
    
    def ask_unanswerable(self, conversations):
        conversations = copy.deepcopy(conversations)
        meta_msg = []
        prev_questions_str = json.dumps(self.repeat_ref_dict["unanswerable"], indent=2) if self.repeat_ref_dict["unanswerable"] else "None"
        conversations.append({"role": "user", "content": UNANSWERABLE_CONV_PROMPT1.format(prev_questions_str)})
        message = self.llm_chat_conv.chat(conversations, None)
        meta_msg.append(message)
        conversations.append({"role": "assistant", "content": message})
        conversations.append({"role": "user", "content": UNANSWERABLE_CONV_PROMPT2})
        message = self.llm_chat_conv.chat(conversations, None)
        meta_msg.append(message)
        conversations.append({"role": "assistant", "content": message})
        prev_questions_str = json.dumps(self.repeat_ref_dict["unanswerable"], indent=2) if self.repeat_ref_dict["unanswerable"] else "None"
        conversations.append({"role": "user", "content": UNANSWERABLE_CONV_PROMPT3.format(prev_questions_str)})
        message = self.llm_chat_conv.chat(conversations, parse_json)
        # Default gt to standard message for unanswerable questions
        if "gt" not in message:
            message["gt"] = "I can't answer the question because the object doesn't exist."
        meta_msg.append(message)
        # Track this question to prevent repetition
        self.repeat_ref_dict["unanswerable"].append(message.get("question", ""))
        return message, meta_msg
    
    def ask_adversarial(self, conversations):
        conversations = copy.deepcopy(conversations)
        meta_msg = []
        prev_questions_str = json.dumps(self.repeat_ref_dict["adversarial"], indent=2) if self.repeat_ref_dict["adversarial"] else "None"
        conversations.append({"role": "user", "content": ADVERSARIAL_CONV_PROMPT1.format(prev_questions_str)})
        message = self.llm_chat_conv.chat(conversations, parse_json)
        # Default gt to 'No' for adversarial questions
        if "gt" not in message:
            message["gt"] = "No"
        meta_msg.append(message)
        # Track this question to prevent repetition
        self.repeat_ref_dict["adversarial"].append(message.get("question", ""))
        return message, meta_msg
    
    def run(self):
        """Run conversation generation with general investigation goal.
        
        Returns:
            List of conversation results to save
        """
        to_save = []
        
        # Use all nodes from the image for investigation
        selected_nodes = copy.deepcopy(self.all_nodes)
        random.shuffle(selected_nodes)  # Randomize order
        
        print("-" * 50, f"Starting general investigation", "-" * 50)
        print("Total nodes to investigate: ", len(selected_nodes))
        print("-" * 100)
        conversations = [
            {"role": "system", "content": CONV_SYSTEM_PROMPT.format(self.image_info)}
        ]
        to_save_i = []
        switch_history = []
        # Reset conversation history for conversation-aware models
        if hasattr(self.eval_func, 'reset'):
            self.eval_func.reset()
        r = 0
        while True:
            if r == 0:
                type_id = 1 # first round always ask regular question
            else:
                type_id = self.switch(conversations, switch_history, r)
                # If END is selected, run switch a second time to confirm
                if type_id == 5:
                    print("END selected, confirming...")
                    type_id_confirm = self.switch(conversations, switch_history, r)
                    if type_id_confirm == 5:
                        print("END confirmed, ending conversation")
                        switch_history.append(type_id)
                        break
                    else:
                        print(f"END not confirmed, proceeding with type {type_id_confirm}")
                        type_id = type_id_confirm
            switch_history.append(type_id)
            if type_id == 5:
                break
            if type_id == 1:
                if len(selected_nodes) == 0:
                    continue
                print("asking regular question")
                message_evaluator = self.ask_regular(conversations, selected_nodes)
            elif type_id == 2:
                print("asking follow-up question")
                message_evaluator = self.ask_follow_up(conversations)
            elif type_id == 3:
                print("asking adversarial question")
                message_evaluator, adv_meta_msg = self.ask_adversarial(conversations)
            elif type_id == 4:
                print("asking unanswerable question")
                message_evaluator, una_meta_msg = self.ask_unanswerable(conversations)
            question = message_evaluator["question"]
            gt = message_evaluator["gt"]
            conversations.append({"role": "assistant", "content": question})
            image_file = self.case["image"]
            output = eval_func(image_file=image_file, query=question)
            # Handle tuple return for conversation-aware models
            if isinstance(output, tuple):
                output = output[0]
            # Ensure output is a string and clean it
            if not isinstance(output, str):
                output = str(output)
            output = output.lower().strip()
            # Remove or escape problematic characters that could corrupt JSON
            # Replace null bytes and control characters (except newlines/tabs)
            output = ''.join(c if ord(c) >= 32 or c in '\n\t' else ' ' for c in output)
            conversations.append({"role": "user", "content": output})
            print("-" * 50, f"round {r}", "-" * 50)
            print(f"examiner question: {question}")
            print(f"examiner gt: {gt}")
            print(f"vlm model: {output}")
            print("-" * 100)
            r += 1
            saved_message = {"round_id": r, "prompt": question, "response":output, "q_type": Q_TYPE_MAPPING[type_id], "gt": gt}
            if type_id == 3:
                saved_message["meta_msg"] = adv_meta_msg
            elif type_id == 4:
                saved_message["meta_msg"] = una_meta_msg
            to_save_i.append(saved_message)
            if r > args.max_rounds:
                print("reached max rounds")
                break
        sample_to_save = copy.deepcopy(self.case)
        sample_to_save["conversations"] = to_save_i
        del sample_to_save["image"]
        to_save.append(sample_to_save)
        return to_save
            


def load_cache(cache_file):
    """Load existing results from cache file if it exists.
    
    Returns:
        list: List of cached conversation results
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            print(f"Loaded {len(cached_data)} cached conversation results from {cache_file}")
            return cached_data
        except json.JSONDecodeError as e:
            print(f"Warning: Cache file {cache_file} is corrupted (JSON decode error: {e})")
            print("Attempting to load backup or starting fresh...")
            # Try to load from temp file if it exists (might be a complete write)
            temp_file = cache_file + '.tmp'
            if os.path.exists(temp_file):
                try:
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    print(f"Loaded {len(cached_data)} results from temp file, restoring...")
                    os.replace(temp_file, cache_file)  # Restore from temp
                    return cached_data
                except:
                    pass
            return []
        except Exception as e:
            print(f"Warning: Error loading cache file {cache_file}: {e}")
            return []
    else:
        return []

def save_cache(cache_file, data):
    """Save results to cache file using atomic write to prevent corruption."""
    try:
        # Use atomic write: write to temp file first, then rename
        # This ensures the original file is only replaced if write succeeds
        temp_file = cache_file + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        
        # Atomic rename - only succeeds if temp file was written completely
        os.replace(temp_file, cache_file)
        print(f"Saved {len(data)} results to cache file {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save to cache file {cache_file}: {e}")
        traceback.print_exc()
        # Clean up temp file if it exists
        temp_file = cache_file + '.tmp'
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

def get_sample_id(sample):
    """Generate a unique identifier for a sample based on its content."""
    # Use image_id as the primary identifier
    return sample.get("image_id", "unknown")

def is_sample_cached(sample_id, cached_data):
    """Check if a sample has already been processed.
    
    Args:
        sample_id: The image_id to look for
        cached_data: List of all cached conversation results
    
    Returns:
        bool: True if sample is already cached
    """
    for item in cached_data:
        if item.get("image_id") == sample_id:
            return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--icls', type=str, default=None)
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--cache_file', type=str, default=None, 
                       help='Cache file to store/load intermediate results for resuming')
    parser.add_argument('--max_rounds', type=int, default=50)
    args = parser.parse_args()
    
    # This script is for models without conversation history, so never enable conversation
    args.use_conversation = False
    
    # Set default cache file if not provided
    if args.cache_file is None:
        args.cache_file = args.outfile.replace('.json', '_cache.json')

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    # need to figure out how to eval on different models
    eval_func = load_model(args)
    
    samples = load_data(args)
    
    llm_chat_conv = LLMChat(model_name="gpt-4o")  # For conversation generation
    
    # Initialize cache and resume functionality
    to_save = []
    cached_data = []
    
    # Load existing cache if provided
    if args.cache_file:
        cached_data = load_cache(args.cache_file)
        to_save.extend(cached_data)
        print(f"Loaded {len(cached_data)} cached samples")
    
    print("starting conversation with model...")
    total_samples = len(samples)
    completed_samples = len(cached_data)
    
    for i, sample in enumerate(tqdm.tqdm(samples, desc="Processing samples")):
        sample_id = get_sample_id(sample)
        
        # Check if sample is already cached
        if is_sample_cached(sample_id, cached_data):
            print(f"Skipping sample {i+1}/{total_samples} (already processed): {sample_id}")
            continue
        
        print(f"Processing sample {i+1}/{total_samples}: {sample_id}")
        
        try:
            eval_sample = EvalSample(sample, llm_chat_conv, eval_func)
            conv = eval_sample.run()
            to_save.extend(conv)
            completed_samples += 1
            
            # Save cache incrementally if cache file is provided
            if args.cache_file:
                save_cache(args.cache_file, to_save)
                print(f"Progress: {completed_samples}/{total_samples} samples completed")
                
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            print("Continuing with next sample...")
            traceback.print_exc()
            continue
    
    # Final save to output file using atomic write
    temp_file = args.outfile + '.tmp'
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(to_save, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        os.replace(temp_file, args.outfile)  # Atomic rename
    except Exception as e:
        print(f"Error saving output file: {e}")
        traceback.print_exc()
        # Clean up temp file if it exists
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        raise
    
    print(f"Completed processing {completed_samples}/{total_samples} samples")
    print(f"Results saved to {args.outfile}")
    if args.cache_file:
        print(f"Cache saved to {args.cache_file}")
