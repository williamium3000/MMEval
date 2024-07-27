CONVERSATION_PROMPT = \
"""
Task: Your task is to evaluate whether a vision-language model will hallucinate (i.e. provide information contradictory to the image). 
Given the grounding of the image, your task is to perform a series of casual conversations with the model naturally by asking questions or making statements about the image and determine whether the model is hallucinating or not in its response.
The conversation is multi-turn and can be open-ended. You need to ask questions based on the history of the conversations. 

Requirements:
1. You need to perform a casual conversation with the model naturally by asking questions or making statements about the image.
2. The conversation is multi-turn and you need to ask questions based on the history of the conversations.
3. When you try to ask questions, you need to consider several factors: 
COVERAGE: the whole conversation should cover all the information provided to you about the image and ask questions that cover as many details as possible. For example, if the model responses fail to cover some specific object or attributes of the objects, you should cover this in the subsequent conversation.
4. You can start with a general question such as 'Please provide a brief description of the image' or 'What is the main object in the image?' and then ask more specific questions based on the response. 
5. Make the conversation as natural as possible. You should act as if you are a human having conversation with the model. You can also do role-playing or act in context, such as asking for assistance in understanding the image.
6. At each turn, you should only provide your part of the conversation and wait for the VLM to respond.
7. If you have asked all the questions and covered all the information about the image, you can end the conversation by outputing "END".
8. Please act as if you are having the conversation directly with the vision-language model. And the the response from the vision-language model will be directly given to you, as if the model is also having the conversation with you.

Provided ground-truth information about the image:
{}

Please respond as if you are having the conversation with the vision-language model. If you want to end the conversation, please output "END" ONLY.
"""

CONVERSATION_PROMPT_GROUND_TRUTH = \
"""
Task: Your task is to evaluate whether a vision-language model will hallucinate (i.e. provide information contradictory to the image). 
Given the grounding of the image, your task is to perform a series of casual conversations with the model naturally by asking questions or making statements about the image and determine whether the model is hallucinating or not.
Each question you ask should have a corresponding ground-truth answer. The conversation is multi-turn and can be open-ended. You need to ask questions and generate ground-truth answers based on the history of the conversations. 

Requirements:
1. You need to perform a casual conversation with the model naturally by asking questions or making statements about the image.
2. The conversation is multi-turn and you need to ask questions and generate ground-truth answers based on the history of the conversations.
3. When you try to ask questions and generate ground-truth answers, you need to consider several factors: 
COVERAGE: the whole conversation should cover all the information provided to you about the image and ask questions that cover as many details as possible. For example, if the model responses fail to cover some specific object or attributes of the objects, you should cover this in the subsequent conversation.
ANSWERABLE: generate a ground-truth answer based on the ground-truth information for each question you ask.
4. You can start with a general question such as 'Please provide a brief description of the image' or 'What is the main object in the image?' and then ask more specific questions based on the response. 
5. Make the conversation as natural as possible. You should act as if you are a human having conversation with the model. You can also do role-playing or act in context, such as asking for assistance in understanding the image.
6. At each turn, you should only provide your part of the conversation and wait for the VLM to respond.
7. If you have asked all the questions and covered all the information about the image, you can end the conversation by outputing "END".
8. Please act as if you are having the conversation directly with the vision-language model. And the the response from the vision-language model will be directly given to you, as if the model is also having the conversation with you.

Format: You should output the question-answer pair in the format of a dictionary. The dictionary contains the two keys: question, ground_truth.
{{"question": <question>, "ground_truth": <ground truth>}}

Examples for question-answer pairs:
Ground-truth information: A black cat with green eyes sitting in the sun.
Ground-truth information: A cat that is staring at the camera.
Ground-truth information: A black cat sitting in a field of grass.

-----Example 1-----
{{"question": "Please provide a brief description of the image.", "ground_truth": "A black cat with green eyes sitting in the sun."}}
-----Example 2-----
{{"question": "What is the cat doing?", "ground_truth": "The cat is staring at the camera."}}
-----Example 3-----
{{"question": "What is the color of the white cat's eyes?", "ground_truth": "Green."}}
-----Example 4-----
{{"question": "How many cats are sitting in the image?", "ground_truth": "One."}}
-----Example 5-----
{{"question": "How is the weather in the image?", "ground_truth": "It is a sunny dat."}}

Provided ground-truth information about the image:
{}

You only respond in the format described above. If you want to end the conversation, please output "END" ONLY.
"""

CONVERSATION_PERSONA_SELECTION_PROMPT = \
"""
Task: Given the grounding of the image and a list of provided personas with their objectives, your task is to select suitable personas for a conversational agent to perform a series of casual conversations with a vision-language model in hallucination detection task.

Requirements:
1. Choose five personas that are suitable for a conversational agent to simulate in casual conversations with a vision-language model. These personas should be the most likely to see this image and elicit hallucinations from the vision-language model.
2. For each selected persona, create an initial question for the vision-language model. The question should align with the traits of the persona and be answerable based only on the image.

Provided persona list:
[
    {{"persona": "Enthusiast of visual content, exploring diverse images with curiosity and interest."}},
    {{"persona": "Passionate researcher delving into visual data patterns and relations between objects."}},
    {{"persona": "Storyteller captivated by narratives within photographs."}},
    {{"persona": "Researcher examining societal perceptions through visual media."}},
    {{"persona": "Art curator dedicated to exploring cultural and historical contexts."}},
    {{"persona": "Marketing analyst evaluating visual branding strategies."}},
    {{"persona": "Hobbyist photographer seeking creative inspiration and technical advice."}},
    {{"persona": "Nature lover documenting biodiversity and conservation through images."}},
    {{"persona": "Gamer exploring realism and visuals in virtual environments."}},
    {{"persona": "Social media influencer focused on engaging visual content."}},
    {{"persona": "Legal researcher analyzing visual evidence in legal proceedings."}},
    {{"persona": "Blind people need assistance to understand the images."}}
]

Provided ground-truth information about the image:
{}

Output format: You should output the selected personas with a list of dictionaries, where each dictionary represents one persona, one context and a prompt. Each dictionary contains the following keys: persona, objective, question.
[
    {{"persona": <persona desc with less than 50 words>, "objective": <context desc>, "question": <question>}},
    ...
]
"""

PERSONA_PROMPT = \
"""
Task: Given the grounding of the image, generate a persona with a context that will be simulated by a conversational agent. The goal is to engage in a casual conversation with a vision-language model and induce it to hallucinate.

Requirements: 
1. Create a brief description of such a person with context. The person is an image viewer.
2. Your description should be short but precise (less than 50 words), only including the most representative traits of the persona and context (e.g., characteristic, motivation and identity).
3. The created persona have an objective. This objective should require the persona to query information grounded in the image and be based on the traits of the created persona.
4. Create an initial question for a vision-language model based on these descriptions. The question should be answered based on the image only.

Format: You should generate a list of dictionaries, where each dictionary represents one persona, one context and a prompt. Each dictionary contains the following keys: persona, objective, question.
[
    {{"persona": <persona desc with less than 50 words>, "objective": <context desc>, "question": <question>}},
    ...
]

Examples for generated personas:
[    
    {{"persona": "As a detective using SenseVision, I look for insights into visual scenes to help solve cases.", "objective": "I aim to uncover relationships and actions in crime scene photos.", "question": "What can you infer from this image of the crime scene?"}},
    {{"persona": "As a teacher, I use it to associate textual descriptions with specific parts of an image for my lessons.", "objective": "I want to help students understand how textual descriptions map to visual elements.", "question": "Which part of the image corresponds to 'the blue car'?"}},
    {{"persona": "As a historian, I want to know the contexts of historical images.", "objective": "I aim to deduce plausible historical events and contexts from visual materials.", "question": "What historical events might this photograph be depicting?"}},
    {{"persona": "As a content creator, I develop narratives based on visual content.", "objective": "I aim to craft engaging stories that combine both visual and textual elements.", "question": "What narrative can you create based on these sequential images?"}},
    {{"persona": "As a blind person, I want to navigate with images for orientation.", "scenario": "I need assistance in identifying landmarks and obstacles from images for navigation purposes.", "question": "How to exit the room?"}}
]

Provided ground-truth information about the image:
{}

Please generate 5 personas, objectives and questions. You MUST only respond in the format as described above. DO NOT RESPOND WITH ANYTHING ELSE.
#Begin
"""

# CONVERSATION_PERSONA_PROMPT = \
# """
# Task: Your task is to simulate a person in context using the given persona with its objective and evaluate whether a vision-language model will hallucinate (i.e. provide information contradictory to the image) on images under such context.
# Given the grounding of the image and the provided persona and objective, your task is to act as the given persona under the context to perform a series of casual conversations with the model naturally by asking questions or making statements about the image.
# The conversation is multi-turn and can be open-ended and you need to ask questions based on the history of the conversations.
#
# Requirements:
# 1. You need to perform a casual conversation with the model naturally by asking questions or making statements about the image under the given context following the provided persona.
# 2. The conversation is multi-turn and you need to ask questions based on the history of the conversations.
# 3. When you try to ask questions, you need to consider several factors:
# FOCUS: the questions asked or the statements made must match the traits of the selected personas.
# COVERAGE: the whole conversation should cover all the information provided to you about the image and ask questions that cover as many details as possible. For example, if the model responses fail to cover some specific object or attributes of the objects, you should cover this in the subsequent conversation.
# NATURALNESS: make the conversation as natural as possible. You should act as if you are a human having conversation with the model. You can also do role-playing or act in context, such as asking for assistance in understanding the image.
# 4. Make the conversation as natural as possible. You should act as if you are a human having conversation with the model.
# 5. At each turn, you should only provide your part of the conversation and wait for the VLM to respond.
# 6. If you have asked all the questions and covered all the information about the image, you can end the conversation by outputing "END".
# 7. Please act as if you are having the conversation directly with the vision-language model, acting as the provided persona in the given context. And the the response from the vision-language model will be directly given to you, as if the model is also having the conversation with you.
# 8. You should NEVER provide the ground-truth information about the image to the vision-language model being evaluated (i.e. DON'T mention any given ground-truth information actively in your question to the model).
#
# Bad example:
# Ground-truth information: The image contains traffic light or an umbrella in the street.
# You: Could you detail any other objects or environmental features in the image that might provide context to the scene, such as a traffic light or an umbrella?
# Reason: you should never mention ground-truth information such as a traffic light or an umbrella, which may give a hint to the vlm.
#
# Persona: {}
#
# Objective: {}
#
# Initial question: {}
#
# Provided ground-truth information about the image:
# {}
#
# Please start with the initial question and respond as if you are having the conversation with the vision-language model. If you want to end the conversation, please output "END" ONLY.
# """

CONVERSATION_PERSONA_PROMPT = \
"""
Task: Your task is to simulate a person in context using the given persona with its objective and evaluate whether a vision-language model will hallucinate (i.e. provide information contradictory to the image) on images under such context. 
Given the grounding of the image and the provided persona and objective, your task is to act as the given persona under the context to perform a series of casual conversations with the model naturally by asking questions or making statements about the image.
The conversation is multi-turn and can be open-ended and you need to ask questions based on the history of the conversations. 

Requirements:
1. You need to perform a casual conversation with the model naturally by asking questions or making statements about the image under the given context following the provided persona.
2. The conversation is multi-turn and you need to ask questions based on the history of the conversations.
3. When you try to ask questions, you need to consider several factors: 
FOCUS: the questions asked or the statements made must match the traits of the selected personas.
NATURALNESS: make the conversation as natural as possible. You should act as if you are a human having conversation with the model. You can also do role-playing or act in context, such as asking for assistance in understanding the image.
4. Make the conversation as natural as possible. You should act as if you are a human having conversation with the model.
5. At each turn, you should only provide your part of the conversation and wait for the VLM to respond.
6. If you have asked all the questions, you can end the conversation by outputing "END".
7. Please act as if you are having the conversation directly with the vision-language model, acting as the provided persona in the given context. And the the response from the vision-language model will be directly given to you, as if the model is also having the conversation with you.
8. You should NEVER provide the ground-truth information about the image to the vision-language model being evaluated (i.e. DON'T mention any given ground-truth information actively in your question to the model).

Bad example:
Ground-truth information: The image contains traffic light or an umbrella in the street.
You: Could you detail any other objects or environmental features in the image that might provide context to the scene, such as a traffic light or an umbrella?
Reason: you should never mention ground-truth information such as a traffic light or an umbrella, which may give a hint to the vlm.

Persona: {}

Objective: {}

Initial question: {}

Provided ground-truth information about the image:
{}

Please start with the initial question and respond as if you are having the conversation with the vision-language model. If you want to end the conversation, please output "END" ONLY.
"""

CONVERSATION_PERSONA_CONTEXT_PROMPT = \
"""
Task: Your task is to simulate a person in context and evaluate whether a vision-language model will hallucinate (i.e. provide information contradictory to the image) on images under such context. 
Given the grounding of the image, your task is to act as the given persona under the context to perform a series of casual conversations with the model naturally by asking questions or making statements about the image.
The conversation is multi-turn and can be open-ended and you need to ask questions based on the history of the conversations. 

Requirements:
1. You need to perform a casual conversation with the model naturally by asking questions or making statements about the image under the given context following the provided persona.
2. The conversation is multi-turn and you need to ask questions based on the history of the conversations.
3. When you try to ask questions, you need to consider several factors: 
COVERAGE: the whole conversation should cover all the information provided to you about the image and ask questions that cover as many details as possible. For example, if the model responses fail to cover some specific object or attributes of the objects, you should cover this in the subsequent conversation.
NATURALNESS: make the conversation as natural as possible. You should act as if you are a human having conversation with the model. You can also do role-playing or act in context, such as asking for assistance in understanding the image.
4. Make the conversation as natural as possible. You should act as if you are a human having conversation with the model.
5. At each turn, you should only provide your part of the conversation and wait for the VLM to respond.
6. If you have asked all the questions and covered all the information about the image, you can end the conversation by outputing "END".
7. Please act as if you are having the conversation directly with the vision-language model, acting as the provided persona in the given context. And the the response from the vision-language model will be directly given to you, as if the model is also having the conversation with you.
8. You should NEVER provide the ground-truth information about the image to the vision-language model being evaluated (i.e. DON'T mention any given ground-truth information actively in your question to the model).

Bad example:
Ground-truth information: The image contains traffic light or an umbrella in the street.
You: Could you detail any other objects or environmental features in the image that might provide context to the scene, such as a traffic light or an umbrella?
Reason: you should never mention ground-truth information such as a traffic light or an umbrella, which may give a hint to the vlm.

Persona: {}

Context: {}

Provided ground-truth information about the image:
{}

Start the conversation with the context. Please respond as if you are having the conversation with the vision-language model. If you want to end the conversation, please output "END" ONLY.
"""

CONVERSATION_UQ_PROMPT = \
"""
Task: Your task is to evaluate whether a vision-language model will hallucinate (i.e. provide information contradictory to the image). 
Given the grounding of the image, your task is to perform a series of casual conversations with the model naturally by asking questions or making statements about the image, and determine whether the model is hallucinating or not in its response.
The conversation is multi-turn and can be open-ended, with questions based on the history of the conversations. The main type of questions in this conversation are unanswerable questions.

Requirements:
1. You need to perform a casual conversation with the model naturally by asking questions or making statements about the image.
2. The conversation is multi-turn and you need to ask questions based on the history of the conversations.
3. When you try to ask questions, you need to consider several factors: 
FOCUS: the questions asked or the statements made must match the traits of the selected personas.
COVERAGE: the whole conversation should cover all the information provided to you about the image and ask questions that cover as many details as possible. For example, if the model responses fail to cover some specific object or attributes of the objects, you should cover this in the subsequent conversation.
FOCUS: the main type of questions in this conversation are unanswerable questions.
NATURALNESS: make the conversation as natural as possible. You should act as if you are a human having conversation with the model. You can also do role-playing or act in context, such as asking for assistance in understanding the image.
4. To generate unanswerable questions, replace the objects and adjectives in the ground-truth image information with their antonyms. Exclude existence and yes-no questions (e.g. “Are there any ...?”).
5. You can start with a general question such as 'Please provide a brief description of the image' or 'What is the main object in the image?' and then ask more specific questions based on the response. 
6. At each turn, you should only provide your part of the conversation and wait for the VLM to respond.
7. If you have asked all the questions and covered all the information about the image, you can end the conversation by outputing "END".
8. Please act as if you are having the conversation directly with the vision-language model. And the the response from the vision-language model will be directly given to you, as if the model is also having the conversation with you.

Examples for unanswerable question:
-----Example 1-----
Ground-truth information: A black cat with green eyes sitting in the sun.
You: What is the color of the white cat's eyes?
Reason: The information in this question contradicts the ground-truth information.
-----Example 2-----
Ground-truth information: A cat that is staring at the camera.
You: What is the dog doing?
Reason: The information in this question contradicts the ground-truth information.
-----Example 3-----
Ground-truth information: A black cat sitting in a field of grass.
You: What animal is sitting on the chair?
Reason: The information in this question contradicts the ground-truth information.

Provided ground-truth information about the image:
{}

Please respond as if you are having the conversation with the vision-language model. If you want to end the conversation, please output "END" ONLY.
"""

CONVERSATION_ADVERSARIAL_PROMPT = \
"""
Task: Your task is to evaluate whether a vision-language model will hallucinate (i.e. provide information contradictory to the image). 
Given the grounding of the image, your task is to perform a series of casual conversations with the model naturally by asking questions or making statements about the image, and determine whether the model is hallucinating or not in its response.
The conversation is multi-turn and can be open-ended, with questions based on the history of the conversations. The main type of questions in this conversation are adversarial questions.

Requirements:
1. You need to perform a casual conversation with the model naturally by asking questions or making statements about the image.
2. The conversation is multi-turn and you need to ask questions based on the history of the conversations.
3. When you try to ask questions, you need to consider several factors: 
COVERAGE: the whole conversation should cover all the information provided to you about the image and ask questions that cover as many details as possible. For example, if the model responses fail to cover some specific object or attributes of the objects, you should cover this in the subsequent conversation.
DIVERSITY: the main type of questions in this conversation are adversarial questions designed to deliberately elicit hallucination.
NATURALNESS: make the conversation as natural as possible. You should act as if you are a human having conversation with the model. You can also do role-playing or act in context, such as asking for assistance in understanding the image.
4. To generate adversarial questions, use the words frequently co-occurring with the object in the ground-truth information. The question should be answerable based on the provided ground-truth information.
5. You can start with a general question such as 'Please provide a brief description of the image' or 'What is the main object in the image?' and then ask more specific questions based on the response. 
6. At each turn, you should only provide your part of the conversation and wait for the VLM to respond.
7. If you have asked all the questions and covered all the information about the image, you can end the conversation by outputing "END".
8. Please act as if you are having the conversation directly with the vision-language model. And the the response from the vision-language model will be directly given to you, as if the model is also having the conversation with you.

Examples for adversarial question:
-----Example 1-----
Ground-truth information: A black cat with green eyes sitting in the sun.
Related word: fur
You: Describe the rainbow-colored fur of the cat in this image.
Reason: The cat has black fur, making the mention of rainbow-colored fur implausible.
-----Example 2-----
Ground-truth information: A cat that is staring at the camera.
Related word: game
You: What game is the cat playing while staring at the camera?
Reason: The cat is only described as staring at the camera, not playing a game.
-----Example 3-----
Ground-truth information: A black cat sitting in a field of grass.
Related word: run
You: How many cats are running in the image?
Reason: Since the ground truth only mentions one cat sitting in a field of grass, the model might incorrectly identify multiple running cats or additional animals.

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

Format: You should return the scene graph in json format, listing objects with attributes, and relations. 
Please note that instances key correspond to a list of instances. If there are two instance of the same class / object, then both should be included in the instances list.
```json
{{
    "instances": [
        {{"name": <name of the entity>, "attributes": <attributes of the entity>}},
        ...
    ],
    "relations": [
        {{"subject": <subject of the relation>, "relation": <relation>, "object": <object of the relation>}},
        ...
    ]
}}
```

Provided text to be parsed:
{}
`
Please think step by step and parse the scene graph from the text carefully.
You MUST only respond in the format as described below. DO NOT RESPOND WITH ANYTHING ELSE.
#Begin
'''