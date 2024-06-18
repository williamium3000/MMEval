from dyna.utils import call_chatgpt
from dyna.prompt import CONVERSATION_PROMPT
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from  infer.infer_llava import eval_model
import os

case = load_coco2017(324158)

CONVERSATION_PROMPT = CONVERSATION_PROMPT.format(case)
conversations = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": CONVERSATION_PROMPT}
]

model_path = "liuhaotian/llava-v1.5-13b"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, get_model_name_from_path(model_path)
    )

turn_cnt = 0
while True:
    turn_cnt += 1
    message_evaluator = call_chatgpt(conversations).content
    if "END" in message_evaluator:
        break
    
    conversations.append({"role": "assistant", "content": message_evaluator})
    image_file = os.path.join("data/coco/val2017", case["file_name"])
    output = eval_model(model_name, tokenizer, model, image_processor, context_len, type('Args', (), {
                            "model_path": model_path,
                            "model_base": None,
                            "model_name": model_name,
                            "query": message_evaluator,
                            "conv_mode": None,
                            "image_file": image_file,
                            "sep": ",",
                            "load_in_8bit": False,
                            "load_in_4bit": False,
                            "temperature": 0.0,  # set as 0.0 for reproceduce
                            "top_p": None,
                            "num_beams": 1,
                            "max_new_tokens": 512
                        })())
    output = output.strip().replace(".", '').lower()
    conversations.append({"role": "user", "content": output})
    print(f"Turn {turn_cnt}: ")
    print("Evaluator: ", message_evaluator)
    print("VLM: ", output)
    print("-------" * 10)
    