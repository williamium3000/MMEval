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

# Point at our local Qwen3 vLLM by default. Override via VALOR_OPENAI_BASE_URL / VALOR_OPENAI_API_KEY.
_BASE_URL = os.environ.get("VALOR_OPENAI_BASE_URL", "http://localhost:8088/v1")
_API_KEY  = os.environ.get("VALOR_OPENAI_API_KEY",  "william")
client = OpenAI(base_url=_BASE_URL, api_key=_API_KEY)
_DEFAULT_MODEL = os.environ.get("VALOR_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

start_marker = ["```json", "```python"]
end_marker = "```"

def llm(prompt, stop=["\n"], model=None):
    if model is None or model == "gpt-5":
        model = _DEFAULT_MODEL
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
