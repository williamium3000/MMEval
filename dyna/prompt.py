CONVERSATION_PROMPT = \
"""
Task: Your task is to have a conversation with a vision-language model regarding the image provided to you. This image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
You need to perform a series of casual conversations with a vision-language model naturally by asking questions or making statements about the image to access the model's perception of the image (whether the model will hallucinate, provide information contradictory to the image or fail to have knowledge regarding some parts or entities in the image).
The conversation is multi-turn and can be open-ended. You need to ask questions based on the history of the conversations. 

Requirements:
1. The conversation is a multi-turn process and your current response should be based on the history of the conversations.
2. At each round, you should only provide your part of the conversation and wait for the model to respond.
3. You should make the conversation as natural as possible and act as if you are a human having conversation directly with the model.
4. COVERAGE: the whole conversation is expected to COVER all the information regarding the image, you should ask questions that cover as many details as possible. If the model responses fail to cover some specific object, attributes or relations in the image, you should cover this in the subsequent conversations.
5. DO NOT DISCLOSE any given image information (captions and bboxes) directly to the model in your conversation. Also, DO NOT mention that the information source is the caption or the bounding box.
6. Focus your conversation on the content of the image. Remember the purpose of the conversation is to assess the model's perception and knowledge of the image.
6. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 
7. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
(1) one can see the content in the image that the question asks about and can answer confidently;
(2) one can determine confidently from the image that it is not in the image.

Bad example:
Ground-truth information: The image contains traffic light or an umbrella in the street.
You: Could you detail any other objects or environmental features in the image that might provide context to the scene, such as a traffic light or an umbrella?
Reason: you should never mention ground-truth information such as a traffic light or an umbrella, which may give a hint to the vlm.

Image information:
{}

You can start the conversation first.
"""

# CONVERSATION_PROMPT_GROUND_TRUTH = \
# """
# Task: Your task is to have a conversation with a vision-language model regarding the image provided to you. This image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
# You need to perform a series of casual conversations with a vision-language model naturally by asking questions or making statements about the image to access the model's perception of the image (whether the model will hallucinate, provide information contradictory to the image or fail to have knowledge regarding some parts or entities in the image).
# Each question you ask should have a corresponding ground-truth answer. The conversation is multi-turn and can be open-ended. You need to ask questions and generate ground-truth answers based on the history of the conversations.
#
# Requirements:
# 1. The conversation is a multi-turn process and your current response should be based on the history of the conversations.
# 2. At each round, you should only provide your part of the conversation and wait for the model to respond.
# 3. You should make the conversation as natural as possible and act as if you are a human having conversation directly with the model.
# 4. COVERAGE: the whole conversation is expected to COVER all the information regarding the image, you should ask questions that cover as many details as possible. If the model responses fail to cover some specific object, attributes or relations in the image, you should cover this in the subsequent conversations.
# 5. ANSWERABLE: generate a ground-truth answer based on the ground-truth information for each question you ask.
# 6. DO NOT DISCLOSE any given image information (captions and bboxes) directly to the model in your conversation. Also, DO NOT mention that the information source is the caption or the bounding box.
# 7. Focus your conversation on the content of the image. Remember the purpose of the conversation is to assess the model's perception and knowledge of the image.
# 8. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing {{"question": "END", "ground_truth": "END"}}.
# 9. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
# (1) one can see the content in the image that the question asks about and can answer confidently;
# (2) one can determine confidently from the image that it is not in the image.
#
# Format: You should output the question-answer pair in the format of JSON which contains two keys: question, ground_truth.
# {{"question": <question>, "ground_truth": <ground truth>}}
#
# Examples for question-answer pairs:
# Image information:
# A black cat with green eyes sitting in the sun.
# A cat that is staring at the camera.
# A black cat sitting in a field of grass.
#
# -----Example 1-----
# {{"question": "Please provide a brief description of the image.", "ground_truth": "A black cat with green eyes sitting in the sun."}}
# -----Example 2-----
# {{"question": "What is the cat doing?", "ground_truth": "The cat is staring at the camera."}}
# -----Example 3-----
# {{"question": "What is the color of the white cat's eyes?", "ground_truth": "Green."}}
# -----Example 4-----
# {{"question": "How many cats are sitting in the image?", "ground_truth": "One."}}
# -----Example 5-----
# {{"question": "How is the weather in the image?", "ground_truth": "It is a sunny day."}}
#
# Image information:
# {}
#
# You can start the conversation first. If you want to end the conversation, please output {{"question": "END", "ground_truth": "END"}}.
# """


CONVERSATION_PROMPT_GROUND_TRUTH = \
"""
Task: Your task is to have a conversation with a human regarding the image provided to you. This image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
You need to perform a series of casual conversations with a human naturally by asking questions or making statements about the image to access the human's perception of the image (whether the model will hallucinate, provide information contradictory to the image or fail to have knowledge regarding some parts or entities in the image).
Each question you ask should have a corresponding ground-truth answer. The conversation is multi-turn and can be open-ended. You need to ask questions and generate ground-truth answers based on the history of the conversations. 

Requirements:
1. The conversation is a multi-turn process and your current response should be based on the history of the conversations.
2. At each round, you should only provide your part of the conversation and wait for the human to respond.
3. You should make the conversation as natural as possible and act as if you are a human having conversation directly with the human.
4. COVERAGE: the whole conversation is expected to COVER all the information regarding the image, you should ask questions that cover as many details as possible. If the human responses fail to cover some specific object, attributes or relations in the image, you should cover this in the subsequent conversations.
5. ANSWERABLE: generate a ground-truth answer based on the ground-truth information for each question you ask.
6. DO NOT DISCLOSE any given image information (captions and bboxes) directly to the model in your conversation. Also, DO NOT mention that the information source is the caption or the bounding box.
7. Focus your conversation on the content of the image. Remember the purpose of the conversation is to assess the human's perception and knowledge of the image.
8. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing {{"question": "END", "ground_truth": "END"}}. 
9. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
(1) one can see the content in the image that the question asks about and can answer confidently;
(2) one can determine confidently from the image that it is not in the image.

Format: You should output the question-answer pair in the format of JSON which contains two keys: question, ground_truth.
{{"question": <question>, "ground_truth": <ground truth>}}

Examples for question-answer pairs:
Image information:
A black cat with green eyes sitting in the sun.
A cat that is staring at the camera.
A black cat sitting in a field of grass.

-----Example 1-----
{{"question": "Please provide a brief description of the image.", "ground_truth": "A black cat with green eyes sitting in the sun."}}
-----Example 2-----
{{"question": "What is the cat doing?", "ground_truth": "The cat is staring at the camera."}}
-----Example 3-----
{{"question": "What is the color of the white cat's eyes?", "ground_truth": "Green."}}
-----Example 4-----
{{"question": "How many cats are sitting in the image?", "ground_truth": "One."}}
-----Example 5-----
{{"question": "How is the weather in the image?", "ground_truth": "It is a sunny day."}}

Image information:
{}

You can start the conversation first. If you want to end the conversation, please output {{"question": "END", "ground_truth": "END"}}.
"""

CONVERSATION_PERSONA_SELECTION_PROMPT = \
"""
Task: Given the grounding of the image and a list of provided personas with their objectives, your task is to select suitable personas for a conversational agent to perform a series of casual conversations with a vision-language model in hallucination detection task.
This image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.

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
    {{"persona": "Blind people needing assistance to understand the images."}},
    {{"persona": "Teacher using images to teach young children how to write."}},
    {{"persona": "Security analyst identifying unusual activities or objects in given images."}}
]

Image information:
{}

Output format: You should output the selected personas with a list of dictionaries, where each dictionary represents one persona, one context and a prompt. Each dictionary contains the following keys: persona, objective, question.
[
    {{"persona": <persona desc with less than 50 words>, "objective": <context desc>, "question": <question>}},
    ...
]
"""

PERSONA_PROMPT = \
"""
Task: Given the grounding of the image, your task is to generate a persona with a context that will be simulated by a conversational agent. The goal is to engage in a casual conversation with a vision-language model and access the model's perception of the image.
This image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.

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
    {{"persona": "As a detective, I look for insights into visual scenes to help solve cases.", "objective": "I aim to uncover relationships and actions in crime scene photos.", "question": "What can you infer from this image of the crime scene?"}},
    {{"persona": "As a teacher, I use it to associate textual descriptions with specific parts of an image for my lessons.", "objective": "I want to help students understand how textual descriptions map to visual elements.", "question": "Which part of the image corresponds to 'the blue car'?"}},
    {{"persona": "As a historian, I want to know the contexts of historical images.", "objective": "I aim to deduce plausible historical events and contexts from visual materials.", "question": "What historical events might this photograph be depicting?"}},
    {{"persona": "As a content creator, I develop narratives based on visual content.", "objective": "I aim to craft engaging stories that combine both visual and textual elements.", "question": "What narrative can you create based on these sequential images?"}},
    {{"persona": "As a blind person, I want to navigate with images for orientation.", "scenario": "I need assistance in identifying landmarks and obstacles from images for navigation purposes.", "question": "How to exit the room?"}}
]

Image information:
{}

Please generate 5 personas, objectives and questions. You MUST only respond in the format as described above. DO NOT RESPOND WITH ANYTHING ELSE.
"""

CONVERSATION_PERSONA_PROMPT = \
"""
Task: Your task is to simulate a person in context using the given persona with its objective and have a conversation with a vision-language model regarding the image provided to you. This image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
You need to act as the given persona under the context and perform a series of casual conversations with a vision-language model naturally by asking questions or making statements about the image to access the model's perception of the image (whether the model will hallucinate, provide information contradictory to the image or fail to have knowledge regarding some parts or entities in the image).
The conversation is multi-turn and can be open-ended. You need to ask questions based on the history of the conversations. 

Requirements:
1. The conversation is a multi-turn process and your current response should be based on the history of the conversations.
2. At each round, you should only provide your part of the conversation and wait for the model to respond.
3. You should make the conversation as natural as possible and act as if you are a human having conversation directly with the model.
4. FOCUS: the questions asked or the statements made must match the traits of the selected personas. You can do role-playing or act in context, such as asking for assistance in understanding the image.
5. DO NOT DISCLOSE any given image information (captions and bboxes) directly to the model in your conversation. Also, DO NOT mention that the information source is the caption or the bounding box.
6. Focus your conversation on the content of the image. Remember the purpose of the conversation is to assess the model's perception and knowledge of the image.
7. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 
8. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
(1) one can see the content in the image that the question asks about and can answer confidently;
(2) one can determine confidently from the image that it is not in the image.

Bad example:
Ground-truth information: The image contains traffic light or an umbrella in the street.
You: Could you detail any other objects or environmental features in the image that might provide context to the scene, such as a traffic light or an umbrella?
Reason: you should never mention ground-truth information such as a traffic light or an umbrella, which may give a hint to the vlm.

Persona: {}

Objective: {}

Initial question: {}

Image information:
{}

You can start the conversation first.
"""

CONVERSATION_UQ_PROMPT = \
"""
Task: Your task is to have a conversation with a vision-language model regarding the image provided to you. This image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
You need to perform a series of casual conversations with a vision-language model naturally by asking questions or making statements about the image to access the model's perception of the image (whether the model will hallucinate, provide information contradictory to the image or fail to have knowledge regarding some parts or entities in the image).
The conversation is multi-turn and can be open-ended. The main type of questions in this conversation are unanswerable questions. You need to ask questions based on the history of the conversations.

Requirements:
1. The conversation is a multi-turn process and your current response should be based on the history of the conversations.
2. At each round, you should only provide your part of the conversation and wait for the model to respond.
3. You should make the conversation as natural as possible and act as if you are a human having conversation directly with the model.
4. COVERAGE: the whole conversation is expected to COVER all the information regarding the image, you should ask questions that cover as many details as possible. If the model responses fail to cover some specific object, attributes or relations in the image, you should cover this in the subsequent conversations.
5. FOCUS: the main type of questions in this conversation are unanswerable questions.
6. To generate unanswerable questions, replace the objects and adjectives in the ground-truth image information with their antonyms. Exclude existence and yes-no questions (e.g. “Are there any ...?”).
7. DO NOT DISCLOSE any given image information (captions and bboxes) directly to the model in your conversation. Also, DO NOT mention that the information source is the caption or the bounding box.
8. Focus your conversation on the content of the image. Remember the purpose of the conversation is to assess the model's perception and knowledge of the image.
9. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 
10. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
(1) one can see the content in the image that the question asks about and can answer confidently;
(2) one can determine confidently from the image that it is not in the image.

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

Image information:
{}

You can start the conversation first.
"""

CONVERSATION_ADVERSARIAL_PROMPT = \
"""
Task: Your task is to have a conversation with a vision-language model regarding the image provided to you. This image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
You need to perform a series of casual conversations with a vision-language model naturally by asking questions or making statements about the image to access the model's perception of the image (whether the model will hallucinate, provide information contradictory to the image or fail to have knowledge regarding some parts or entities in the image).
The conversation is multi-turn and can be open-ended. The main type of questions in this conversation are adversarial questions. You need to ask questions based on the history of the conversations.

Requirements:
1. The conversation is a multi-turn process and your current response should be based on the history of the conversations.
2. At each round, you should only provide your part of the conversation and wait for the model to respond.
3. You should make the conversation as natural as possible and act as if you are a human having conversation directly with the model.
4. COVERAGE: the whole conversation is expected to COVER all the information regarding the image, you should ask questions that cover as many details as possible. If the model responses fail to cover some specific object, attributes or relations in the image, you should cover this in the subsequent conversations.
5. FOCUS: the main type of questions in this conversation are adversarial questions designed to deliberately elicit hallucination.
6. To generate adversarial questions, use the words frequently co-occurring with the object in the ground-truth information. The question should be answerable based on the provided ground-truth information.
7. DO NOT DISCLOSE any given image information (captions and bboxes) directly to the model in your conversation. Also, DO NOT mention that the information source is the caption or the bounding box.
8. Focus your conversation on the content of the image. Remember the purpose of the conversation is to assess the model's perception and knowledge of the image.
9. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 
10. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
(1) one can see the content in the image that the question asks about and can answer confidently;
(2) one can determine confidently from the image that it is not in the image.

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

Image information:
{}

You can start the conversation first.
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
You are an excellent scene graph parser that can parse the scene graph from language accurately.

Task: You will be given a paragraph of text, describing the objects, attributes of objects and relations between objects in an image. Your task is to parse the scene graph from the texts and return a scene graph in json format.

Requirements:
1. You need parse sentence by sentence and update the scene graph step by step based on the information in the sentence.
2. For each sentence, you should first identify the concrete objects. Then identify attributes of these objects and  the relations between objects.
Examples of concrete objects: man, dog, tree, car, etc. Examples of non-concrete objects (abstract): atmosphere, setting, scene, etc. This is only example, you should consider beyond this and only select the concrete objects.
Examples of attributes: color, size, shape, quantity, etc. You should consider beyond this.
Relation between objects includes but not limited to spatial relation such as on, under, near, and interaction relation such as holding, eating, size comparison such as bigger, smaller, etc. You should consider beyond this.
3. After you parse the objects, attributes and relation in one sentence, you should update the scene graph following the rules:
- for each object, you should first determine whether this object already exists in the scene graph or not using contexts, attributes of this object and its relation with other objects. If it exists, you should update the attributes and relations of this object. If it does not exist, you should add this object to the scene graph. You should particularly notice different instances of the same class.
- for each attribute, you should determine whether the subject already exists in the scene graph or not. If it exists, you should update the attribute. If it does not exist, you should first add the obejct and this attribute to the scene graph.
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