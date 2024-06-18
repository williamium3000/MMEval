CONVERSATION_PROMPT = \
"""
Task: Your task is to evaluate whether a vision-language model will hallucinate (i.e. provide information contradictory to the image) on images. 
Given the detailed information (grounding of the image), your task is to perform a series of casual conversations with the model naturally by asking questions or making statements about the image and determine whether the model is hallucinating or not in its response.
The conversation is multi-turn and can be open-ended and you need to ask questions based on the history of the conversations. 

Requirement:
1. You need to perform a casual conversation with the model naturally by asking questions or making statements about the image. You should provide a correct answer to the question.
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