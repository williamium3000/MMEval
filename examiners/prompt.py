CONV_MODEL_PERSPECTIVE_PROMPT = \
"""
Task: Your task is to have a conversation with a vision-language model regarding a given image and assess whether it will hallucinate (i.e. provide information contradictory to the image) during the conversation.
This given image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
You need to perform a series of casual conversations with the vision-language model naturally and ask questions or make statements about the given detailed information of the image. The responses from the vision-language model will be later to used to assess whether the model is faithful to the image.
The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is a multi-turn process and your current response should be based on the history of the conversations.
2. At each round, you should only provide your part of the conversation and wait for the human to respond.
3. You should make the conversation as natural as possible and act as if you are a human having conversation directly with another human.
4. COVERAGE: the whole conversation is expected to COVER all the information regarding the image, you should ask questions that cover as many details as possible. If the human responses fail to cover some specific object, attributes or relations in the image, you should cover this in the subsequent conversations.
5. DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention that the information source is the caption or the bounding box.
6. Focus your conversation on the content of the image. Remember the purpose of the conversation is to assess the human's perception and knowledge of the image.
7. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 
8. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers:
    a. one can see the content in the image that the question asks about and can answer confidently.
    b. one can determine confidently from the image that it is not in the image.

Bad examples:
Image information:
"A man is in a kitchen making pizzas.",
"Man in apron standing on front of oven with pans and bakeware",
"A baker is working in the kitchen rolling dough.",
"A person standing by a stove in a kitchen.",
"A table with pies being made and a person standing near a wall with pots and pans hanging on the wall."

You: Are there any smaller items, like a bottle or specific ingredient like broccoli or carrots, you can identify in the scene?
Reason: you should not mention the objects in the image directly. For example, you can ask: Are there any smaller items you can identify in the scene?

You: The organization definitely adds to the efficiency. I also noticed a bottle on the table. What do you think it could contain\u2014perhaps some oil, vinegar, or maybe a seasoning for the dishes?
Reason: This question cannot be answered confidently from the given image information. You should only ask questions that have definite answers.

You: What do you think about the rack or wall with pots and pans hanging around the workspace? Does it contribute to the vibe of the kitchen?
Reason: first, do not mention detailed information from the image directly. Second, this question cannot be answered confidently from the given image information. You should only ask questions that have definite answers.


Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""

CONV_COVERAGE_PROMPT = \
"""
Task: Your task is to have a conversation with a human regarding the image provided to you. This image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
You need to perform a series of casual conversations with a human naturally by asking questions or making statements about the image to assess human's perception of the image (whether the human will hallucinate, provide information contradictory to the image or fail to have knowledge regarding some parts or entities in the image).
The conversation is multi-turn and can be open-ended and you need to ask questions based on the history of the conversations. 

Requirements:
1. The conversation is a multi-turn process and your current response should be based on the history of the conversations.
2. At each round, you should only provide your part of the conversation and wait for the human to respond.
3. You should make the conversation as natural as possible and act as if you are a human having conversation directly with another human.
4. COVERAGE: the whole conversation is expected to COVER all the information regarding the image, you should ask questions that cover as many details as possible. If the human responses fail to cover some specific object, attributes or relations in the image, you should cover this in the subsequent conversations.
5. DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention that the information source is the caption or the bounding box.
6. Focus your conversation on the content of the image. Remember the purpose of the conversation is to assess the human's perception and knowledge of the image.
7. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Bad examples:
Image information:
"A man is in a kitchen making pizzas.",
"Man in apron standing on front of oven with pans and bakeware",
"A baker is working in the kitchen rolling dough.",
"A person standing by a stove in a kitchen.",
"A table with pies being made and a person standing near a wall with pots and pans hanging on the wall."

You: Are there any smaller items, like a bottle or specific ingredient like broccoli or carrots, you can identify in the scene?
Reason: you should not mention the objects in the image directly. For example, you can ask: Are there any smaller items you can identify in the scene?

You: The organization definitely adds to the efficiency. I also noticed a bottle on the table. What do you think it could contain\u2014perhaps some oil, vinegar, or maybe a seasoning for the dishes?
Reason: This question cannot be answered confidently from the given image information. You should only ask questions that have definite answers.

You: What do you think about the rack or wall with pots and pans hanging around the workspace? Does it contribute to the vibe of the kitchen?
Reason: first, do not mention detailed information from the image directly. Second, this question cannot be answered confidently from the given image information. You should only ask questions that have definite answers.


Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""

CONV_COVERAGE_PROMPT_CERTAINTY = \
"""
Task: Your task is to have a conversation with a human regarding the image provided to you. This image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
You need to perform a series of casual conversations with a human naturally by asking questions or making statements about the image to assess human's perception of the image (whether the human will hallucinate, provide information contradictory to the image or fail to have knowledge regarding some parts or entities in the image).
The conversation is multi-turn and can be open-ended and you need to ask questions based on the history of the conversations. 

Requirements:
1. The conversation is a multi-turn process and your current response should be based on the history of the conversations.
2. At each round, you should only provide your part of the conversation and wait for the human to respond.
3. You should make the conversation as natural as possible and act as if you are a human having conversation directly with another human.-
4. COVERAGE: the whole conversation is expected to COVER all the information regarding the image, you should ask questions that cover as many details as possible. If the human responses fail to cover some specific object, attributes or relations in the image, you should cover this in the subsequent conversations.
5. DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention that the information source is the caption or the bounding box.
6. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 
7. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers:
    a. one can see the content in the image that the question asks about and can answer confidently.
    b. one can determine confidently from the image that it is not in the image.

Bad examples:
Image information:
"A man is in a kitchen making pizzas.",
"Man in apron standing on front of oven with pans and bakeware",
"A baker is working in the kitchen rolling dough.",
"A person standing by a stove in a kitchen.",
"A table with pies being made and a person standing near a wall with pots and pans hanging on the wall."

You: Are there any smaller items, like a bottle or specific ingredient like broccoli or carrots, you can identify in the scene?
Reason: you should not mention the objects in the image directly. For example, you can ask: Are there any smaller items you can identify in the scene?

You: The organization definitely adds to the efficiency. I also noticed a bottle on the table. What do you think it could contain\u2014perhaps some oil, vinegar, or maybe a seasoning for the dishes?
Reason: This question cannot be answered confidently from the given image information. You should only ask questions that have definite answers.

You: What do you think about the rack or wall with pots and pans hanging around the workspace? Does it contribute to the vibe of the kitchen?
Reason: first, do not mention detailed information from the image directly. Second, this question cannot be answered confidently from the given image information. You should only ask questions that have definite answers.


Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""

CONV_COVERAGE_PROMPT_CERTAINTY_WITH_ANSWER = \
"""
Task: Your task is to have a conversation with a human regarding the image provided to you. This image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
You need to perform a series of casual conversations with a human naturally by asking questions or making statements about the image to assess human's perception of the image (whether the human will hallucinate, provide information contradictory to the image or fail to have knowledge regarding some parts or entities in the image).
The conversation is multi-turn and can be open-ended and you need to ask questions based on the history of the conversations. 

Requirements:
1. The conversation is a multi-turn process and your current response should be based on the history of the conversations.
2. At each round, you should only provide your part of the conversation and wait for the human to respond.
3. You should make the conversation as natural as possible and act as if you are a human having conversation directly with another human.-
4. COVERAGE: the whole conversation is expected to COVER all the information regarding the image, you should ask questions that cover as many details as possible. If the human responses fail to cover some specific object, attributes or relations in the image, you should cover this in the subsequent conversations.
5. DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention that the information source is the caption or the bounding box.
6. For each question, you should provide a ground-truth answer that will be later used to assess the human. The generated ground-truth answer should be determined by the given image captions and object bboxs in the image.
7. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
    a. one can see the content in the image that the question asks about and can answer confidently.
    b. one can determine confidently from the image that it is not in the image.
8. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Image information:
{}

Format: for each round of the conversation, you should provide a question and the ground-truth answer to the question in a json dictionary {{"prompt": "<question>", "response": "<ground_truth_answer>"}} ONLY.
If you want to end the conversation, you can output {{"prompt": "END", "response": "END"}} ONLY.

Please respond as if you are having the conversation with the vision-language model directly.
"""




PROMPTS = {
    "coverage_certainty": CONV_COVERAGE_PROMPT_CERTAINTY,
    "coverage": CONV_COVERAGE_PROMPT,
    "coverage_certainty_with_answer": CONV_COVERAGE_PROMPT_CERTAINTY_WITH_ANSWER,
    "coverage_model_perspective": CONV_MODEL_PERSPECTIVE_PROMPT,
}