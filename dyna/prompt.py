CONVERSATION_PROMPT = \
"""
Task: Your task is to evaluate whether a vision-language model will hallucinate (i.e. provide information contradictory to the image) on images. 
Given the detailed information (grounding of the image), your task is to perform a series of casual conversations with the model naturally by asking questions or making statements about the image and determine whether the model is hallucinating or not in its response.
The conversation is multi-turn and can be open-ended and you need to ask questions based on the history of the conversations. 

Requirement:
1. You need to perform a casual conversation with the model naturally by asking questions or making statements about the image.
2. The conversation is multi-turn and you need to ask questions based on the history of the conversations.
3. When you try to ask questions, you need to consider several factors: 
COVERAGE: the whole conversation should cover all the information provided to you about the image and ask questions that cover as many details as possible. For example, if the model responses fail to cover some specific object or attributes of the objects, you should cover this in the subsequent conversation.
4. You can start with a general question such as 'Please provide a brief description of the image' or 'What is the main object in the image?' and then ask more specific questions based on the response. 
5. Make the conversation as natural as possible. You should act as if you are a human having conversation with the model. You can also do role-playing or act in context, such as asking for assistance in understanding the image.
6. At each turn, you should only provide your part of the conversation and wait for the VLM to respond.
7. If you have asked all the questions and covered all the information about the image, you can end the conversation by outputing "END"
Now, please start your conversation with the vision-language model.
8. Please act as if you are having the conversation directly with the vision-language model. And the the response from the vision-language model will be directly given to you, as if the model is also having the conversation with you.
Provided ground-truth information about the image:
{}

Please respond as if you are having the conversation with the vision-language model. If you want to end the conversation, please output "END" ONLY.
"""

PERSONA_PROMPT = \
"""
Task: You are a helpful and precise prompt generator. Given the detailed information (grounding of the image), your task is to is to generate a persona with a context that will be simulated by a conversational agent to induce a vision-language model to hallucinate.
This persona and context will be later to prompt a ChatGPT-like LLM to simulate a person in such context and have a causal conversation with a human regarding this image.

Requirement: create a brief description of such a person with characteristic, background, and preferences and a clear goal with context.

Provided ground-truth information about the image:
{}

Persona and context:
"""

CONVERSATION_PERSONA_PROMPT = \
"""
Task: Your task is to simulate a person in context using the given persona and context information and evaluate whether a vision-language model will hallucinate (i.e. provide information contradictory to the image) on images under such context. 
Given the detailed information (grounding of the image) and the provided persona and context, your task is to perform a series of casual conversations with the model naturally by asking questions or making statements about the image under the given context following the provided persona.
The conversation is multi-turn and can be open-ended and you need to ask questions based on the history of the conversations. 

Requirement:
1. You need to perform a casual conversation with the model naturally by asking questions or making statements about the image under the given context following the provided persona.
2. The conversation is multi-turn and you need to ask questions based on the history of the conversations.
3. When you try to ask questions, you need to consider several factors: 
COVERAGE: the whole conversation should cover all the information provided to you about the image and ask questions that cover as many details as possible. For example, if the model responses fail to cover some specific object or attributes of the objects, you should cover this in the subsequent conversation.
4. Make the conversation as natural as possible. You should act as if you are a human having conversation with the model.
5. At each turn, you should only provide your part of the conversation and wait for the VLM to respond.
6. If you have asked all the questions and covered all the information about the image, you can end the conversation by outputing "END"
Now, please start your conversation with the vision-language model.
7. Please act as if you are having the conversation directly with the vision-language model, acting as the provided persona in the given context. And the the response from the vision-language model will be directly given to you, as if the model is also having the conversation with you.

Persona and context: 
{}

Provided ground-truth information about the image:
{}

Please respond as if you are having the conversation with the vision-language model. If you want to end the conversation, please output "END" ONLY.
"""

GRADER_PROMPT = \
'''
You are an impartial and objective judge with expertise in assessing the hallucination in a paragraph of text.

Task: Your task is to evaluate whether a paragraph of text contains hallucination given the ground-truth grounding information, i.e. contains information contradictory to the ground-truth grounding information. Ground-truth grounding information will be given to you that presents all the information in the image you know.
Hallucination, in this context, refers to a situation where the generated response (a paragraph of text) that includes information not present or implied in the ground-truth grounding information of an image. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.

Requirements:
1. You must carefully judge from three aspects, including the object, attributes and relations between objects in the image and check whether any of these aspects are hallucinated in the text.
"Object" specifically refers to whether the objects in the texts actually exists in the image exist and whether the quantity of objects in the response conflicts with the grounding information given.
"Attributes" specifically refer to whether the attributes of objects (e.g. color, position, action) in the response conflicts with the attribute in the images or in the given grounding information.
"Relation" specifically refers to whether the relation or interaction between objects described in the response conflicts with the relation or interaction between objects in the image or in the given grounding information.
2. You will also receive detection results from the expert model. The object detection expert model will provide detected entity names along with their bounding box information in the image. When deriving position relationships between entity instances, try to also use the bounding boxes information, which are represented as [x1, y1, x2, y2] with floating numbers ranging from 0 to 1. These values correspond to the top left x1, top left y1, bottom right x2, and bottom right y2. The scene text expert model will provide detected specific text along with their bounding box information in the image. As long as there is a conflict between a single letter in the scene text and the text information required in the claim, it’s considered a hallucination.
3. You will need to carefully check word by word, sentence by sentence to detect all hallucination. For EACH SENTENCE, YOU MUST RETURN THE JUDGMENT RESULTS IN A DICTIONARY to identify whether this sentence is hallucinating (marked as 0), correct (marked as 1) or uncertain (marked as 2), based on the following criteria:
If the sentence conflicts the ground-truth grounding information, you should mark it as 0.
If the sentence is consistent with the ground-truth grounding information, you should mark it as 1.
If the sentence contains information (objects, attributes or relations) that are not present in the ground-truth grounding information, you should mark it as 2.
You should also provide a reasoning and justification for your judgment. You should ground the judgement to the given ground-truth information for justification. For example, you should include the bbox provided in ground-truth grounding information to justify your judgment.

Format: You should return a list of dictionaries, where each dictionary represents the results of one sentence. Each dictionary contains the following keys: sentence, result_type, reason. 
[
    {{"sentence": <the sentence of interest>, "result_type": <the result type, 0 for hallucination, 1 for correct or 2 for uncertain>, "reason": <justification for the judgement> }},
    ...
]

Provided ground-truth information about the image:
{}

Response needed to be evaluated:
{}

You MUST only respond in the format as described below. DO NOT RESPOND WITH ANYTHING ELSE.
#Begin
'''


LSG_PROMPT = \
'''
You are an excelent scene graph parser that can parse the scene graph from language accurately.

Task: You will be given a paragraph of text, describing the objects, attributes of objects and relations between objects in an image. Your task is to parse the scene graph from the texts and return a scene graph in json format.

Requirements:
1. You need parse sentence by sentence and update the scene graph step by step based on the information in the sentence.
2. For each sentence, you should first identify the concrete objects. Then identify attributes of these objects and  the relations between objects.
Examples of concrete objects: man, dog, tree, car, etc. Examples of non-concrete objects (abstract): atmosphere, setting, scene, etc. This is only example, you should consider beyond this and only select the concrete objects.
Examples of attributes: color, size, shape, quantity, etc. You should consider beyond this.
Relation between objects includes but not limited to spatial relation such as on, under, near, and interaction relation such as holding, eating, size comparison such as bigger, smaller, etc. You should consider beyond this.
3. After you parse the objects, attributes and relation in one sentence, you should update the scene graph following the rules:
- for each object, you should first determin whether this object already exists in the scene graph or not using contexts, attributes of this object and its relation with other objects. If it exists, you should update the attributes and relations of this object. If it does not exist, you should add this object to the scene graph. You should particularly notice different instances of the same class.
- for each attribute, you should determin whether the subject already exists in the scene graph or not. If it exists, you should update the attribute. If it does not exist, you should first add the obejct and this attribute to the scene graph.
- for each relation, you should first determine the subject and object exists in the scene graph or not. If they exist, you should update the relation. If they do not exist, you should first add the subject and object to the scene graph, then the relation.
4. You should follow the above steps and parse sentence-by-sentence, and update the scene graph step by step.

Format: You should return the scene graph in json format, listing objects, attributes, and relations.
{{
    "entities": [
        {{"name": <name of the entity>, "attributes": <attributes of the entity>}},
        ...
    ],
    "relations": [
        {{"subject": <subject of the relation>, "relation": <relation>, "object": <object of the relation>}},
        ...
    ]
}}

Provided text to be parsed:
{}

Please think step by step but MAKE SURE to output the result in the given FORMAT.
#Begin
'''