CONV_COVERAGE_PROMPT = \
"""
Task: Your task is to have a conversation with a human regarding the image provided to you. This image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
You need to perform a series of casual conversations with a human naturally by asking questions or making statements about the image to access human's perception of the image (whether the human will hallucinate, provide information contradictory to the image or fail to have knowledge regarding some parts or entities in the image).
The conversation is multi-turn and can be open-ended and you need to ask questions based on the history of the conversations. 

Requirements:
1. The conversation is a multi-turn process and your current response should be based on the history of the conversations.
2. At each round, you should only provide your part of the conversation and wait for the human to respond.
3. You should make the conversation as natural as possible and act as if you are a human having conversation directly with another human.-
4. COVERAGE: the whole conversation is expected to COVER all the information regarding the image, you should ask questions that cover as many details as possible. If the human responses fail to cover some specific object, attributes or relations in the image, you should cover this in the subsequent conversations.
5. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Image information:
{}

You can start the conversation first.
"""