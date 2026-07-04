# v10context changes

# 1) MUST CHANGE: 2. Goal: Identify a specific action, objective, or task that "I" am trying to accomplish, which should be coherent with the scene described.
# 2) context be more high-level glimpses without leaking too many observatory details.


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

CONTEXT_PROMPT = \
"""
Your task is to create a realistic scenario in which the given image is situated in the first-person view, i.e. you should imagine the image depicts your sight at the environment. This context should incorporate a background setting, the characters or objects involved, and a specific goal or objective that is relevant to the image. The context must be plausible, align with real-world experiences, and directly connect with the depicted elements in the image.

The image will be described through a list of objects, their attributes, and their spatial relationships, with each object represented by a set of coordinates in the image. These coordinates (x1, y1, x2, y2) will range from 0 to 1, corresponding to the top-left and bottom-right corners of each object.


========================
CORE CONSTRAINTS
========================
1) First-person framing:
   - Write as if the image is what "I" am currently seeing in front of me.
2) Avoid non-visual / non-inductable human interaction goals:
   - If humans are present, you may include them as part of the scene, but the goal must NOT require learning about their demographic information (names, intentions, etc.). 
3) Grounding in image instances:
    - Each context MUST involves multiple instances, attributes or relations from the image.
    - Given sufficient diversity, you should generate contexts that naturally involve as many instances, attributes or relations from the image as possible.
4) Diversity without redundancy:
   - The two contexts must be meaningfully different (different emphasis on objects/relations), not just rephrases of the same scenario.
     
Instructions:
1. Contextualization: Develop a background scenario that is logical and directly relevant to the visual elements in the image. The background should describe the setting, time, and possible situation in which these objects or characters might exist.
2. Goal: Identify a specific action, objective, or task that "I" am trying to accomplish, which should be coherent with the scene described.
3. Diversity: For the given image, you should generate several different and diverse contexts.

Some GOOD Examples:
```json
[
    {{
        "background": "A bustling city street during rush hour, with pedestrians walking past stores and cars honking in traffic. The image depicts the first-person view of the character.",
        "goal": "The character is trying to catch a bus before it leaves."
    }},
    {{
        "background": "An open-plan corporate office during a busy afternoon, with cubicles neatly separated by dividing screens and personal photos decorating the workspace walls. The image depicts the first-person view of the character.",
        "goal": "The character is trying to send an email to his boss."
    }}, 
    {{
        "background": "A bright and inviting kitchen featuring wooden cabinetry, a cozy dining area, and fresh fruit adding a vibrant touch. The image depicts the first-person view of the character",
        "goal": "The character is hungry and tries to eat something."
    }},
    {{
        "background": "A modest bathroom showing signs of wear, featuring basic fixtures, white tiles, and a shower area needing repairs.",
        "goal": "The character just woke up and wanted to wash his face."
    }},
    {{
        "background": "A modern kitchen with stainless steel appliances, wooden cabinets, and a countertop filled with various cooking utensils. ",
        "goal": "The character is blind and stumbed into the kitchen. He is trying to find his way out of the kitchen."
    }}
]

Image information:
{}

Please generate two contexts based on the image information. Please make sure the contexts involves a few objects and instances in the image but stays as high-level glimpses without leaving too many observatory details. Make sure the contexts are diverse and not redundant.

========================
WHAT TO PRODUCE
========================
Return EXACTLY a JSON list with TWO dictionaries. Each dictionary must contain:
- "background": <50 words, first-person situation, plausible real-world setting>
- "goal": a concrete objective that can be progressed by asking image-grounded questions

Output format (STRICT):
```json
[
  {{"background": "...", "goal": "..."}},
  {{"background": "...", "goal": "..."}}
]
```
You MUST only respond in the format as described above. DO NOT RESPOND WITH ANYTHING ELSE.
"""


SELECT_CONTEXT_NODES_PROMPT_SYSTEM = \
"""
you will be give an image and a context (background and goal). Then given image will be presented to you as a list of objects with attributes and relation of these objects. Each objects will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Based on the image and context, you need to select the object nodes that are most relevant to the context, which will later be used to generate the evaluation question on this image and context.
You can select one or multiple object nodes, but make sure they are relevant to the context or can be used to generate a evaluation question that fits the context.

Respond format: You should return a list of selected object node in the order of their relevancy with the context. The first object node should be the most relevant object node to the context.
you can explain, reason or perform chain-of-thought to select the object nodes. However, you MUST eventually provide a list of selected object_id in the json format as described below:
```json
[
    "object_id_1", "object_id_2", "object_id_3", ...
]
```
"""

SELECT_CONTEXT_NODES_PROMPT = \
"""
Image information:
{}

COntext:
background: {}, goal: {}

Please select the object nodes that are most relevant to the context in the order of their relevancy with the context. You can explain, reason or perform chain-of-thought to select the object nodes. However, you MUST respond the selected object_id in the json format as described above.
"""



SWITCH_PROMPT = \
"""Based on the given conversation history, please decide which type of question to ask next. Please choose one of the following types of questions:

1. **Regular questions** – directly related to the image and context.
   *Example*: In an office setting, the goal of the conversation is to send an email to his boss. you can ask questions like "is the monitor turned on?", "where is the power button of the computer?", "I want to type in the email, what should I look for?" etc.

2. **Follow-up questions** – follow up, confirm or interrogate the model’s last response.

3. **Adversarial questions** – inquire about plausible but absent objects that commonly co-occur with visible ones in the image.
   *Example*: If the image shows a cake without utensils, you may ask: *“Can I use the knife on the table to cut the cake?”* or *"Is there a folk?"*

4. **Unanswerable questions** – ask about question that cannot be answered.
   *Example*: if the image depicts a cake on the table without any utensil or people eating the cake, you can ask "What utensil is the man using to cut the cake?". This is an unanswerable question because you cannot answer with an utensil but rather you should answer "There are not any man in the image eating the cake".

5. **End the conversation** – output "END" to end the conversation.

Given previous asked question types: {}, try to ask a diverse types of questions.

YOU CAN ONLY SELECT ONE OF THE ABOVE FIVE TYPES OF QUESTIONS !!
Respond format: you can explain, reason or perform chain-of-thought to select the next question type. 
However, you MUST eventually provide a digit between 1 and 4 to indicate the next question type:
```json
{{"type": type_id}}
```
"""

CONV_SYSTEM_PROMPT = \
"""Your task is to **simulate a multi-round conversation with a vision-language model (VLM) about a given image**, in order to test whether the model hallucinates (i.e., produces responses inconsistent with the image) or remains faithful to it.

You will have multiple rounds of conversations with the model. In each round, you will be provided with:

* **An image** (represented as a list of objects, their attributes, and relationships).
* **Bounding-box coordinates** for each object, given as `(x1, y1, x2, y2)` in normalized values between 0 and 1, corresponding to top-left and bottom-right corners.

Your role is to **carry out a natural, open-ended, human-like conversation** with the model, asking questions about the image in the given context.

Requirements

1. **Multi-turn conversation**:

   * Ask one question per turn, then wait for the model’s response.
   * Incorporate both the image, context, and dialogue history when forming your question.

2. **Natural, human-like tone**:

   * Speak as if conversing casually with another person.
   * Avoid mechanical or scripted phrasing.

4. **No disclosure of metadata**:

   * Do not reveal or reference bounding boxes, object lists, captions, or the source of your information.
   * Ask question as if you are looking at the image with NO access to the bounding boxes, object lists, captions.

5. **Handling mistakes**:

   * Do not correct the model if they make mistakes. 
   * However, you may ask follow-ups to probe or interrogate its response.

6. **Question style**:

   * Prefer open-ended questions over yes/no questions.
   * Ask diverse, non-redundant questions.
   * Only ask questions with **definite answers** (either clearly present in the image or confidently absent).

7. **Conversation ending**:

   * If the dialogue has naturally run its course, output **“END”** (and nothing else).

Image information:
{}


Context:
background: {}, goal: {}
"""

REGULAR_CONV_PROMPT = \
"""You are tasked with generating a *regular question* grounded in the provided image and context.

* **Target node:** `{}`
* **Context:** background: `{}`, goal: `{}`

**Instructions:**

* Ask a natural, conversational question about the specified node.
* The question must be strongly relevant to both the image and the context (background + goal).
* Avoid irrelevant, generic, or out-of-character questions.
* Keep the question consistent with the role of someone actively engaged in the situation.
* Generate the ground-truth answer based on the given image content along the question.

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
"""Based on the given image, context, and the conversation history, please ask a follow-up question about the model's last turn of conversation and the corresponding ground-truth answer.
You should ask the question as if you are having a conversation with the model. You should also following the above requirements.
Please respond in the following format:
```json
{{"question": "xxxx", "gt": "xxxx"}},
```
Please respond with the question and the ground-truth answer ONLY. DO NOT respond with anything else.
"""

ADVERSARIAL_CONV_PROMPT1 = \
"""Based on the given image, context, and the conversation history, please ask an adversarial question about a plausible but absent object, attribute or relation that commonly co-occur with visible ones in the image.
Follow the procedure: first generate a plausible but absent or incorrect object, attribute or relation that commonly co-occur with visible ones in the image in the form of a json dict, then ask the adversarial question based on the generated hallucinated object, attribute or relation.

Requirements:
Your generated hallucinated object, attribute or relation and the final adversarial question should be consistent with the context:
background: {}, goal: {}
Your generated hallucinated object, attribute or relation should not be present in the image but should be highly co-occur with image content.
Your generated adversarial question should be consistent with the conversation history.

Now, please generate a plausible but absent or incorrect object, attribute or relation that commonly co-occur with visible ones in the image in the form of a json dict.
Examples:
If there is a blue banana on the table, you can ask about the color of the banana.
```json
{{"names": "banana", "attributes": "blue"}},
```
If there is cake on the table, you can ask whether there is a knife in the image.
```json
{{"names": "knife"}},
```
or the relation between the banana and the table.
```json
{{"names": "banana", "relations": "on the table"}},
```
"""

ADVERSARIAL_CONV_PROMPT2 = \
"""Now you should ask the adversarial question based on the generated hallucinated object, attribute or relation and the corresponding ground-truth answer.
Please respond in the following format:
```json
{{"question": "xxxx", "gt": "xxxx"}},
```
You should ask a question as if you are having a conversation with the model. Please respond with the question and the ground-truth answer ONLY. DO NOT respond with anything else.
"""

UNANSWERABLE_CONV_PROMPT1 = \
"""Based on the given image, context, and conversation history, generate an **unanswerable question**.

Here is the context:
background: {}, goal: {}

An *unanswerable question* refers to a query that cannot be answered using the provided information because it introduces a plausible but absent or incorrect object, attribute, or relation.
For example, if the image shows only a cake on the table, ask *“What utensil is the man using to cut the cake?”* is unanswerableble since no man is present.

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

4) **Consistency with context and conversation**
   - The question must be consistent with the context (background + goal) and conversation history, even though it is unanswerable.

Good Unanswerable Question Templates (Examples):
- Definite reference + attribute detail:
  * {"question": "What color is the handle of the knife on the table?", "gt": "No knife is visible on the table in the image."}
- Definite reference + relation detail:
  * {"question": "Which side of the cake is the fork placed on—left or right?", "gt": "No fork is visible near the cake in the image."}
- Definite reference + human interaction detail (only if the image has no humans; this makes it unanswerable):
  * {"question": "What utensil is the man using to cut the cake?", "gt": "No man is visible in the image cutting the cake."}

Bad (DISALLOWED) Examples (these would overlap with ADVERSARIAL):
- "Do you see any knife we could use?" (non-presuppositional)
- "If there is a fork, where might it be?" (conditional)

Output Requirement:
Respond in the following json format:
```json
{"question": "xxxx", "gt": "xxxx"}
```

Respond with **the question and gt ONLY**. Do not include explanations, commentary, or any additional text.
"""

Q_TYPE_MAPPING = {
    1: "regular",
    2: "follow-up",
    3: "adversarial",
    4: "unanswerable",
    5: "end",
}
class EvalSample:
    def __init__(self, case, llm_chat, eval_func):
        self.case = case
        self.image_info = format_case_vg(case) if args.dataset == "vg" else format_case_coco(case)
        self.scene_graph_data = SceneGraphData.from_dict(case)

        self.llm_chat = llm_chat
        self.eval_func = eval_func
        self.conversations = []

    def switch(self, conversations, swicth_history):
        conversations = copy.deepcopy(conversations)
        conversations.append(
            {"role": "user", "content": SWITCH_PROMPT.format(swicth_history)})
        type_id = self.llm_chat.chat(conversations, parse_json)['type']
        return type_id
    
    def ask_regular(self, conversations, selected_nodes, context):
        conversations = copy.deepcopy(conversations)
        sampled_node = selected_nodes.pop(0)
        conversations.append({"role": "user", "content": REGULAR_CONV_PROMPT.format(sampled_node, context["background"], context["goal"])})
        message = self.llm_chat.chat(conversations, parse_json)
        
        return message
    
    def ask_follow_up(self, conversations, context):
        conversations = copy.deepcopy(conversations)
        conversations.append({"role": "user", "content": FOLLOW_UP_CONV_PROMPT})
        message = self.llm_chat.chat(conversations, parse_json)
        return message
    
    def ask_unanswerable(self, conversations, context):
        conversations = copy.deepcopy(conversations)
        meta_msg = []
        conversations.append({"role": "user", "content": UNANSWERABLE_CONV_PROMPT1.format(context["background"], context["goal"])})
        message = self.llm_chat.chat(conversations, None)
        meta_msg.append(message)
        conversations.append({"role": "assistant", "content": message})
        conversations.append({"role": "user", "content": UNANSWERABLE_CONV_PROMPT2})
        message = self.llm_chat.chat(conversations, None)
        meta_msg.append(message)
        conversations.append({"role": "assistant", "content": message})
        conversations.append({"role": "user", "content": UNANSWERABLE_CONV_PROMPT3})
        message = self.llm_chat.chat(conversations, parse_json)
        meta_msg.append(message)
        return message, meta_msg
    
    def ask_adversarial(self, conversations, context):
        conversations = copy.deepcopy(conversations)
        meta_msg = []
        conversations.append({"role": "user", "content": ADVERSARIAL_CONV_PROMPT1.format(context["background"], context["goal"])})
        message = self.llm_chat.chat(conversations, None)
        meta_msg.append(message)
        conversations.append({"role": "assistant", "content": message})
        conversations.append({"role": "user", "content": ADVERSARIAL_CONV_PROMPT2})
        message = self.llm_chat.chat(conversations, parse_json)
        meta_msg.append(message)
        return message, meta_msg
    
    def generate_context(self, case):
        image_info = format_case_vg(case) if args.dataset == "vg" else format_case_coco(case)
        conversations = [
            {"role": "system", "content": "You are a expert in generating realistic and diverse contexts for images. You excel at understanding the image content and predicting the possible scenarios and context in which the image might be situated."},
            {"role": "user", "content": CONTEXT_PROMPT.format(image_info).strip()}
        ]
        
        contexts = self.llm_chat.chat(conversations, parse_json)

        return contexts
    
    def select_context_nodes(self, context):
        # let LLM select the object nodes that are most relevant to the context
        # which will be used by the examiner to generate the question
        background = context["background"]
        goal = context["goal"]
        conversations = [
            {"role": "system", "content": SELECT_CONTEXT_NODES_PROMPT_SYSTEM},
            {"role": "user", "content": SELECT_CONTEXT_NODES_PROMPT.format(self.scene_graph_data.image_info.sg, background, goal)}
        ]
        selected_nodes = self.llm_chat.chat(conversations, parse_json)
        selected_object_nodes = [self.scene_graph_data.image_info.sg.get_object_by_id(int(oid)) for oid in selected_nodes]
        return selected_object_nodes
    
    def run(self):
        to_save = []
        contexts = self.generate_context(self.case)
        for context in contexts:
            selected_nodes = self.select_context_nodes(context)
            print("-" * 50, f"context", "-" * 50)
            print(context)
            print("selected nodes: ", selected_nodes)
            print("-" * 100)
            # TODO: can we use ICLs here?
            # sys_prompt = PROMPT.__dict__[args.p_mode]
            # loaded_icls = []
            # if args.icls is not None:
            #     loaded_icls = json.load(open(args.icls))
            
            # ICLs = []
            # for icl in loaded_icls:
            #     firstp = CONV_PROMPT.format(icl["image_info"])
            #     ICLs.append({"role": "user", "content": firstp})
            #     ICLs.extend(icl["conversations"])
            
            conversations = [
                {"role": "system", "content": CONV_SYSTEM_PROMPT.format(self.image_info, context["background"], context["goal"])},
            ]
            to_save_i = []
            switch_history = []
            r = 0
            while True:
                if r == 0:
                    type_id = 1 # first round always ask regular question
                else:
                    type_id = self.switch(conversations, switch_history)
                
                switch_history.append(type_id)
                
                if type_id == 5:
                    break
                
                if type_id == 1:
                    if len(selected_nodes) == 0:
                        continue
                    print("asking regular question")
                    message_evaluator = self.ask_regular(conversations, selected_nodes, context)
                elif type_id == 2:
                    print("asking follow-up question")
                    message_evaluator = self.ask_follow_up(conversations, context)
                elif type_id == 3:
                    print("asking adversarial question")
                    message_evaluator, adv_meta_msg = self.ask_adversarial(conversations, context)
                elif type_id == 4:
                    print("asking unanswerable question")
                    message_evaluator, una_meta_msg = self.ask_unanswerable(conversations, context)
                
                question = message_evaluator["question"]
                gt = message_evaluator["gt"]
                conversations.append({"role": "assistant", "content": question})
            
                image_file = self.case["image"]
                output = eval_func(image_file=image_file, query=question)
                output = output.lower()
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
                to_save_i.append(
                    saved_message
                )
                if r > args.max_rounds:
                    print("reached max rounds")
                    break
            
            sample_to_save = copy.deepcopy(self.case)
            sample_to_save["conversations"] = to_save_i
            sample_to_save["context"] = context
            del sample_to_save["image"]
            to_save.append(sample_to_save)
        
        return to_save
            


def load_cache(cache_file):
    """Load existing results from cache file if it exists."""
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        print(f"Loaded {len(cached_data)} cached results from {cache_file}")
        return cached_data
    else:
        return []

def save_cache(cache_file, data):
    """Save results to cache file."""
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved {len(data)} results to cache file {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save to cache file {cache_file}: {e}")
        traceback.print_exc()

def get_sample_id(sample):
    """Generate a unique identifier for a sample based on its content."""
    # Use image_id as the primary identifier
    return sample.get("image_id", "unknown")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--icls', type=str, default=None)
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--cache_file', type=str, default=None, 
                       help='Cache file to store/load intermediate results for resuming')
    parser.add_argument('--max_rounds', type=int, default=20)
    args = parser.parse_args()
    
    # Set default cache file if not provided
    if args.cache_file is None:
        args.cache_file = args.outfile.replace('.json', '_cache.json')

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    # need to figure out how to eval on different models
    eval_func = load_model(args)
    
    samples = load_data(args)
    
    llm_chat = LLMChat(model_name="gpt-5")
    
    # Initialize cache and resume functionality
    to_save = []
    processed_samples = set()
    
    # Load existing cache if provided
    if args.cache_file:
        cached_data = load_cache(args.cache_file)
        print(cached_data)
        to_save.extend(cached_data)
        # Track which samples have already been processed
        for cached_item in cached_data:
            if "image_id" in cached_item:
                processed_samples.add(cached_item["image_id"])
        print(f"Found {len(processed_samples)} already processed samples in cache")
    
    print("starting conversation with model...")
    total_samples = len(samples)
    processed_count = len(processed_samples)
    
    for i, sample in enumerate(tqdm.tqdm(samples, desc="Processing samples")):
        sample_id = get_sample_id(sample)
        
        # Skip if already processed
        if sample_id in processed_samples:
            print(f"Skipping sample {i+1}/{total_samples} (already processed): {sample_id}")
            continue
        
        print(f"Processing sample {i+1}/{total_samples}: {sample_id}")
        
        try:
            eval_sample = EvalSample(sample, llm_chat, eval_func)
            conv = eval_sample.run()
            to_save.extend(conv)
            processed_samples.add(sample_id)
            processed_count += 1
            
            # Save cache incrementally if cache file is provided
            if args.cache_file:
                save_cache(args.cache_file, to_save)
                print(f"Progress: {processed_count}/{total_samples} samples completed")
                
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            print("Continuing with next sample...")
            traceback.print_exc()
            continue
    
    # Final save to output file
    with open(args.outfile, "w") as f:
        json.dump(to_save, f, indent=4)
    
    print(f"Completed processing {processed_count}/{total_samples} samples")
    print(f"Results saved to {args.outfile}")
    if args.cache_file:
        print(f"Cache saved to {args.cache_file}")
