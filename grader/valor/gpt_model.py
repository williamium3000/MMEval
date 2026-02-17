import os
import glob
import random
import json
import openai
import time
import json
import numpy as np
import os
from openai import OpenAI

client = OpenAI()

start_marker = ["```json", "```python"]
end_marker = "```"

def llm(prompt, stop=["\n"], model="gpt-5"):
    success = False
    output = {}
    while not success:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=prompt,
                # temperature=0,
                # top_p=1,
                # frequency_penalty=0.0,
                # max_tokens=2000,
                # presence_penalty=0.0,
            )
            raw_json_string = None
            for start_mark in start_marker:
                if start_mark in response.choices[0].message.content:
                    start_index = response.choices[0].message.content.find(start_mark)

                    start_of_json_content = start_index + len(start_mark)

                    # Find the closing index of the JSON block (the ending ```)
                    end_index = response.choices[0].message.content.find(end_marker, start_of_json_content)
                    if end_index == -1:
                        raise ValueError("JSON end marker '```' not found after the start marker.")

                    raw_json_string = response.choices[0].message.content[start_of_json_content:end_index].strip()
                    break

            if raw_json_string is None:
                raw_json_string = response.choices[0].message.content

            output = json.loads(raw_json_string)
            success = True
        except Exception as e:
            print(f"Exception: {e}")
            print("Retrying...")
            time.sleep(10)
    time.sleep(1)
    return output
