import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")

import time

client = OpenAI()


def call_chatgpt(messages, model_name="gpt-4o"):
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": message}
    # ]
    while True:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                # temperature=1,
                # max_tokens=256,
                # top_p=1,
                # frequency_penalty=0,
                # presence_penalty=0
            )
            break
        except Exception as e:
            print(e)
            time.sleep(1)

    return completion.choices[0].message.content


