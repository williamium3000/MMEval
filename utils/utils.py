import itertools
from openai import OpenAI
from dotenv import load_dotenv
from nltk.corpus import wordnet as wn

load_dotenv(".env")

client = OpenAI()


def call_chatgpt(messages, model_name="gpt-4o"):
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": message}
    # ]

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    return completion.choices[0].message.content

def call_chatgpt_json(messages, model_name="gpt-4o"):
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content

def get_synset(word):
    syn_list = wn.synonyms(word)
    syn_list = list(itertools.chain.from_iterable(syn_list))
    return [word]+[' '.join(syn.split('_')) for syn in syn_list]
