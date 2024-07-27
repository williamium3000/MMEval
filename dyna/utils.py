import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")

import time

# client = OpenAI()
#
#
# def call_chatgpt(messages, model_name="gpt-4o"):
#     # messages = [
#     #     {"role": "system", "content": "You are a helpful assistant."},
#     #     {"role": "user", "content": message}
#     # ]
#     while True:
#         try:
#             completion = client.chat.completions.create(
#                 model=model_name,
#                 messages=messages,
#                 # temperature=1,
#                 # max_tokens=256,
#                 # top_p=1,
#                 # frequency_penalty=0,
#                 # presence_penalty=0
#             )
#             break
#         except Exception as e:
#             print(e)
#             time.sleep(1)
#
#     return completion.choices[0].message.content


def convert_output_format(input_path, output_path):
    with open(input_path, "r") as input_file:
        sample_list = json.load(input_file)

    revised_result_list = []
    for sample in sample_list:
        # conversation_list = sample["conversation_list"]
        # for conversation in conversation_list:
        #     qa_pair_list = []
        #     for index in range(0, len(conversation["conversation"]), 2):
        #         question = conversation["conversation"][index]["content"]
        #         if index < len(conversation["conversation"])-1:
        #             answer = conversation["conversation"][index+1]["content"]
        #             qa_pair_list.append({"question": question, "answer": answer})
        #
        #     dic = {"image": sample["file_name"],
        #            "visual_info":
        #                {"caption": sample["captions"],
        #                 "bbox": [instance["bbox"] for instance in sample["instances"]],
        #                 "category": [instance["category"] for instance in sample["instances"]]
        #                 },
        #            "qa": qa_pair_list,
        #             "persona": conversation["persona"]
        #            }
        #     revised_result_list.append(dic)

        conversation = sample["conversation"]
        qa_pair_list = []
        for index in range(0, len(conversation), 2):
            question = conversation[index]["content"]
            if index < len(conversation)-1:
                answer = conversation[index+1]["content"]
                qa_pair_list.append({"question": question, "answer": answer})

        dic = {"image": sample["file_name"],
               "visual_info":
                   {"caption": sample["captions"],
                    "bbox": [instance["bbox"] for instance in sample["instances"]],
                    "category": [instance["category"] for instance in sample["instances"]]
                    },
               "qa": qa_pair_list
               }
        revised_result_list.append(dic)

    revised_result_json = json.dumps(revised_result_list)
    with open(output_path, "w") as output_file:
        output_file.write(revised_result_json)


if __name__ == "__main__":
    convert_output_format("../output/conversation.json",
                          "../output/conversation_revised.json")

