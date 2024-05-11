import torch
import os
import argparse
import json
import tqdm
from PIL import Image

from transformers import AutoModelForCausalLM, LlamaTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def eval_model(model, tokenizer, image_file, query, torch_type):
    image = Image.open(image_file).convert('RGB')
    input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])
    inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]] if image is not None else None,
        }
    if 'cross_images' in input_by_model and input_by_model['cross_images']:
        inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

    # add any transformers params here.
    gen_kwargs = {"max_length": 2048,
                    "do_sample": False} # "temperature": 0.9
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        output = tokenizer.decode(outputs[0])
        output = output.split("</s>")[0]
    return output




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
    parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    
    MODEL_PATH = args.from_pretrained
    TOKENIZER_PATH = args.local_tokenizer
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
    if args.bf16:
        torch_type = torch.bfloat16
    else:
        torch_type = torch.float16

    print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

    if args.quant:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_type,
            trust_remote_code=True
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
            load_in_4bit=args.quant is not None,
            trust_remote_code=True
        ).to(DEVICE).eval()
    
    # format:
    # a list of dict
    # minimum keys: image, question
    # [
    #     {
    #         "image": "xxxx",
    #         "question": "xxxxx"},
    #     ...]
    # leave the output key empty
    
    samples = json.load(open(args.infile, "r"))

    for sample in tqdm.tqdm(samples):
        q = sample["question"]
        image_file = os.path.join(args.img_dir, sample["image"])
        output = eval_model(model, tokenizer, image_file, q, torch_type)
        
        output = output.strip().replace(".", '').lower()
        sample["output"] = output
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    json.dump(samples, open(args.outfile, "w"), indent=4)
            



