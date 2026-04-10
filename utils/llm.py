import re
import json
import time
import os

from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(".env")

# Max context tokens for examiner API (truncate if over to avoid context_length_exceeded)
MAX_CONTEXT_TOKENS = int(os.environ.get("LLM_MAX_CONTEXT_TOKENS", "120000"))


def _message_content_length(msg):
    """Rough character count of a message (for token estimate)."""
    content = msg.get("content") if isinstance(msg, dict) else ""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(
            len(item.get("text", "")) if isinstance(item, dict) else len(str(item))
            for item in content
        )
    return len(str(content))


def _estimate_tokens(messages):
    """Estimate token count: ~4 chars per token for mixed text."""
    total_chars = sum(_message_content_length(m) for m in messages)
    return (total_chars // 4) + 1


def _truncate_messages(messages, max_tokens=MAX_CONTEXT_TOKENS):
    """Drop oldest non-system messages until under max_tokens. Keeps system and latest turns."""
    if not messages:
        return messages
    if _estimate_tokens(messages) <= max_tokens:
        return messages
    # Keep system message(s) at start
    system_msgs = []
    rest = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            system_msgs.append(m)
        else:
            rest.append(m)
    # Drop oldest from rest until under limit
    out = list(system_msgs)
    for i in range(len(rest) - 1, -1, -1):
        candidate = system_msgs + rest[i:]
        if _estimate_tokens(candidate) <= max_tokens:
            out = candidate
            break
    else:
        out = system_msgs + rest[-1:] if rest else system_msgs
    if len(out) < len(messages):
        print(f"Truncated conversation (this sample): {len(messages)} -> {len(out)} messages (context limit {max_tokens} tokens)")
    return out

def parse_dict(text):
    pattern = r"{(.*)}"
    match = re.search(pattern, text, re.DOTALL)
    json_text = "{" + (match.group(1) if match else text) + "}"
    return json.loads(json_text)
def parse_json(text):
    pattern = r"```json(.*)```"
    match = re.search(pattern, text, re.DOTALL)
    json_text = match.group(1) if match else text
    return json.loads(json_text)

def parse_code(rsp):
    pattern = r"```python(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text



class LLMChat:
    def __init__(self, model_name=None, patience=7):
        self.patience = patience
        self.model = model_name
        self.client = None
        
        if os.getenv("OPENAI_API_KEY"):
            base_url = os.getenv("OPENAI_BASE_URL")
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url if base_url else None)
        
        if os.getenv("AZURE_OPENAI_KEY"):
            self.client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
                api_key=os.getenv("AZURE_OPENAI_KEY"),  
                api_version="2025-01-01-preview")
            self.model = os.getenv("AZURE_OPENAI_DEPLOYNAME", self.model)
        
        assert self.client is not None, "Neither OPENAI_API_KEY nor AZURE_OPENAI_KEY is set. Please configure one."
        assert self.model is not None, "Model name is not provided."
        
        # log params
        print("*" * 100)
        print(f"calling api model: {self.model}")
        print(f"patience: {patience}")
        print("*" * 100)
        
    def chat(self, messages, parser_fn, response_format=None, verbose=False, **kwargs):
        count = 0
        last_error = None
        while True:
            try:
                messages_to_send = _truncate_messages(messages, MAX_CONTEXT_TOKENS)
                if response_format:
                    response = self._get_structured_response(messages_to_send, response_format, **kwargs)
                else:
                    response = self._get_response(messages_to_send, **kwargs)
                if response is None or (isinstance(response, str) and not response.strip()):
                    raise ValueError("API returned empty or null content")
                if verbose:
                    print(response)
                if parser_fn is None:
                    return response
                else:
                    return parser_fn(response)
            except Exception as e:
                last_error = e
                # Don't retry on 400 Bad Request (e.g. context_length_exceeded) - fail fast
                err_str = str(e).lower()
                if "400" in err_str or "badrequest" in err_str or "context_length" in err_str:
                    print(f"LLM error (non-retryable): {e}")
                    if parser_fn is None:
                        return "END"
                    return None
                count += 1
                if count >= self.patience:
                    print(f"Exceeded patience ({self.patience} retries). Last error: {last_error}")
                    # Return "END" so examiner loop ends cleanly and saves progress (no crash)
                    if parser_fn is None:
                        return "END"
                    return None
                backoff = min(2 ** (count - 1), 60)  # 1, 2, 4, 8, ... max 60s
                print(f"LLM error ({type(e).__name__}): {e}")
                print(f"Retry {count}/{self.patience}, waiting {backoff}s before retry...")
                time.sleep(backoff)
        return None
            

    def _get_response(self, messages, **kwargs):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        content = completion.choices[0].message.content
        if content is None:
            raise ValueError("API returned null content (e.g. refusal or tool use)")
        return content

    def _get_structured_response(self, messages, response_format, **kwargs):
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=response_format,
            **kwargs
        )

        return completion.choices[0].message.parsed
    

if __name__ == "__main__":
    agent = LLMChat("gpt-4o")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "which one is larger? 1.11 or 1.9"}
    ]

    # response = agent.chat(messages, None)
    # print(response)

    class Response(BaseModel):
        response: str
        question_type: str

    response = agent.chat(messages,None, Response)
    print(response)