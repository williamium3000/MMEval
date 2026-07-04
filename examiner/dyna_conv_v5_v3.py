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

Instructions:

1. The context and the goal should natural and plausible with the visual content and setup of the image.
2. The goal is a specific action, objective, or task that the character are trying to accomplish. It should be simple and day-to-day actions.
3. The background should NOT be too detailed description of the scene, but rather as high-level glimpses without too many visual details.
4. The goal should be achievable through asking question about visual content in the image. Avoid goals that involves direct interaction with the humans in the images such as "confirm his name" or "ask how he feels".

The response should be a list of dictionaries, where each dictionary represents one possible context for the image. Each dictionary should contain two keys:
Background: A brief description (fewer than 50 words) of the setting, situation, or scenario that fits the image.
goal: A clear description of the goal or task that is being pursued by the objects/subjects in the image.

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
        "background": "A modern kitchen with stainless steel appliances, wooden cabinets, and a countertop filled with various cooking utensils. ",
        "goal": "The character is blind and stumbed into the kitchen. He is trying to find his way out of the kitchen."
    }}
]
```

Image information:
{}

Please generate two contexts. You MUST only respond in the format as described above. DO NOT RESPOND WITH ANYTHING ELSE.
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

2. **Follow-up questions** – follow up, challenge or interrogate the model’s last response.

3. **Adversarial questions** – inquire about plausible but absent object, attribute or relation that commonly co-occur with visible ones in the image.
   *Example*: If the image shows a cake without utensils, you may ask: *“Is there a knife on the table?”* or *"I want to cut the cake, can you see any knife around?"*

4. **Unanswerable questions** – ask about question that cannot be answered.
   *Example*: if the image depicts a cake on the table without anyone eating the cake, you can ask "What utensil is the man using to cut the cake?". This is an unanswerable question because you cannot answer with an utensil but rather you should answer "There are not any man in the image eating the cake".

5. **End the conversation** – output "END" to end the conversation.


DO NOT encourage a overly long conversation. You should end the conversation if:
1. you have enough information for the goal in context;
2, all relevant visual content has been queried;
3. the conversation has naturally run its course.
4. the conversation is too long and repetitive.

Given previous asked question types: {}, you should balance the four types of questions and try to select a diverse type.

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

5. **Eliciting Hallucinations**:

   * The goal of the conversation is to elicit visual hallucinations from the model.
   * Prefer questions that are likely to cause model to hallucinate in the image.
   * Try to elicit as many visual hallucinations from the model as possible.
   
6. **Handling mistakes**:

   * Do not correct the model if they make mistakes. 
   * However, you may ask follow-ups to further interrogate the model with question relevant to the mistakes to elicit more hallucinations.

7. **Question style**:

   * Prefer open-ended questions over yes/no questions.
   * Ask diverse, non-redundant questions.
   * Only ask questions with **definite answers** (either clearly present in the image or confidently absent).

8. **Conversation ending**:

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
* The question must be strongly relevant to the context goal and grounded on the given target node. Keep the question consistent with the role of someone actively engaged in the situation.
* Avoid irrelevant, generic, or out-of-character questions.
* Ask diverse questions. Avoid questions similar to previous questions in the conversation history.

**Example:**
If the background is an office and the goal is “send an email to the boss,” suitable questions include:

* “Is the monitor turned on?”
* “Where is the computer’s power button?”
* “I want to type the email, what should I look for?”

**Output Requirement:**
Respond with **the question ONLY**. DO NOT include explanations, commentary, or any additional text.
"""

FOLLOW_UP_CONV_PROMPT = \
"""Based on the given image, context, and the conversation history, please ask a follow-up question about the model's last turn of conversation.
Here are some types of follow-up questions:
1. Ask for further challenging details about the previous QA when available;
2. Interrogat the model's confidence in its previous correct answer. For example, you can challenge the model's answer such as "Are you sure about xxxx?" or confronting the model's answer such as "Are you confident about xxx?I think I see xxxx (a wrong answer)".
3. Test consistency. For example, you can ask a similar but rephrased question to test the consistency of the model's answer.

The above is only some example types for your reference. DO NOT limit yourself to the above types. You can ask any follow-up question that is relevant to the previous QA and the image/context.
You should ask a question as if you are having a conversation with the model. You should also follow the above requirements. Please respond with the question ONLY. DO NOT respond with anything else.
"""

ADVERSARIAL_CONV_PROMPT = \
"""Based on the given image, context, and the conversation history, generate an **adversarial question** about a plausible but absent object, attribute or relation that commonly co-occur with visible ones in the image.

An *adversarial question* is question that is likely to cause model to hallucinate in the image. It refers to a query that asks about an object, attribute or relation that does not exist in the image but commonly co-occur with the image and the visual content in the image.

Adversarial questions about an absent but commonly co-occur object generally would be an existence question (i.e. asking whether the object is present in the image). However, you can phrase it to avoid the simple yes/no answer.
examples:
"Is there a knife on the table?"
"I want to cut the cake, can you see any knife around?"
"To help cut the cake on the table, is there any knife and where is it?"

Adversarial questions about an absent but commonly co-occur attribute or relation should be either yes/no or asking directly about the attribute or relation.
examples:
- asking color of the banana:
"Is the banana on the table yellow?"
"What color is the banana on the table?"
-asking the spatial relationship of the cat and the table:
"Is the cat on the table?"
"where is the cat playing?"

You should ask a question as if you are having a conversation with the model. Please respond with the question ONLY. DO NOT respond with anything else.
"""



UNANSWERABLE_CONV_PROMPT1 = \
"""Based on the given image, context, and conversation history, generate an **unanswerable question**.

An *unanswerable question* refers to a query that cannot be answered using the provided information because it introduces a plausible but absent or incorrect object, attribute, or relation.
For example: if the image shows only a cake on the table, asking *“What utensil is the man using to cut the cake?”* is unanswerable since no man is present.

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
* Step 3: ask the unanswerable question about the relation: "eating" between the generated object "man" and the object "cake": “What utensil is the man using to cut the cake?”*

---

**Start Now:**
Complete **Step 1** by generating a plausible but absent or incorrect object in JSON format.
"""

UNANSWERABLE_CONV_PROMPT2 = \
"""Now complete Step 2. Generate a plausible attribute or relation for the hallucinated object you created in Step 1. This relation must link the hallucinated object to one of the real objects present in the image."""

UNANSWERABLE_CONV_PROMPT3 = \
"""Now complete the third step. Please ask an unanswerable question about the attribute or relation of the generated hallucinated object. 

This must be an **UNANSWERABLE (presuppositional trap) question** that **DOES presuppose** the hallucinated object/relation/attribute exists in the image.
The question must sound plausible, but it cannot be answered from the provided image because the key premise is false/absent.

Please respond with the question ONLY. DO NOT respond with anything else.
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
        message = self.llm_chat.chat(conversations, None)
        return message
    
    def ask_follow_up(self, conversations, context):
        conversations = copy.deepcopy(conversations)
        conversations.append({"role": "user", "content": FOLLOW_UP_CONV_PROMPT})
        message = self.llm_chat.chat(conversations, None)
        return message
    
    def ask_unanswerable(self, conversations, context):
        conversations = copy.deepcopy(conversations)
        meta_msg = []
        conversations.append({"role": "user", "content": UNANSWERABLE_CONV_PROMPT1})
        message = self.llm_chat.chat(conversations, None)
        meta_msg.append(message)
        conversations.append({"role": "assistant", "content": message})
        conversations.append({"role": "user", "content": UNANSWERABLE_CONV_PROMPT2})
        message = self.llm_chat.chat(conversations, None)
        meta_msg.append(message)
        conversations.append({"role": "assistant", "content": message})
        conversations.append({"role": "user", "content": UNANSWERABLE_CONV_PROMPT3})
        message = self.llm_chat.chat(conversations, None)
        meta_msg.append(message)
        return message, meta_msg
    
    def ask_adversarial(self, conversations, context):
        conversations = copy.deepcopy(conversations)
        meta_msg = []
        conversations.append({"role": "user", "content": ADVERSARIAL_CONV_PROMPT})
        message = self.llm_chat.chat(conversations, None)
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
                    
                conversations.append({"role": "assistant", "content": message_evaluator})
            
                image_file = self.case["image"]
                output = eval_func(image_file=image_file, query=message_evaluator)
                output = output.lower()
                conversations.append({"role": "user", "content": output})

                print("-" * 50, f"round {r}", "-" * 50)
                print(f"examiner: {message_evaluator}")
                print(f"vlm model: {output}")
                print("-" * 100)
            
                r += 1
                saved_message = {"round_id": r, "prompt": message_evaluator, "response":output, "q_type": Q_TYPE_MAPPING[type_id]}
                if type_id == 3:
                    saved_message["meta_msg"] = adv_meta_msg
                elif type_id == 4:
                    saved_message["meta_msg"] = una_meta_msg
                to_save_i.append(
                    saved_message
                )
                if r >20:
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
