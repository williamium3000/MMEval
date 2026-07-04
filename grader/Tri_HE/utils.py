import time
import json
import os

import spacy
import openai
from openai import RateLimitError as OpenAIRateLimitError
from openai import APIError as OpenAIAPIError
from openai import Timeout as OpenAITimeout


# Setup spaCy NLP
nlp = None

# Setup OpenAI API
openai_client = None

# Setup Claude 2 API
bedrock = None
anthropic_client = None


def sentencize(text):
    """Split text into sentences"""
    global nlp
    if not nlp:
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent for sent in doc.sents]


def split_text(text, segment_len=200):
    """Split text into segments according to sentence boundaries."""
    segments, seg = [], []
    sents = [[token.text for token in sent] for sent in sentencize(text)]
    for sent in sents:
        if len(seg) + len(sent) > segment_len:
            segments.append(" ".join(seg))
            seg = sent
            # single sentence longer than segment_len
            if len(seg) > segment_len:
                # split into chunks of segment_len
                seg = [
                    " ".join(seg[i:i+segment_len])
                    for i in range(0, len(seg), segment_len)
                ]
                segments.extend(seg)
                seg = []
        else:
            seg.extend(sent)
    if seg:
        segments.append(" ".join(seg))
    return segments


def get_openai_model_response(prompt, temperature=0, model='gpt-3.5-turbo', n_choices=1):
    global openai_client
    if not openai_client:
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if not prompt or len(prompt) == 0:
        return None

    while True:
        try:
            if isinstance(prompt, str):
                messages = [{
                    'role': 'user',
                    'content': prompt
                }]
            elif isinstance(prompt, list):
                messages = prompt
            else:
                return None

            res_choices = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    n=n_choices
                ).choices
            if n_choices == 1:
                response = res_choices[0].message.content
            else:
                response = [res.message.content for res in res_choices]

            if response and len(response) > 0:
                return response
        except Exception as e:
            if isinstance(e, OpenAIRateLimitError) or isinstance(e, OpenAIAPIError) or isinstance(e, OpenAITimeout):
                time.sleep(10)
                continue
            print(type(e), e)
            return None
        return None