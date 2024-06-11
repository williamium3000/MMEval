import torch
import os
import argparse
import json
import tqdm
from PIL import Image

from transformers import AutoProcessor, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"


def eval_model(processor, model, image_file, query):
    raw_image = Image.open(image_file).convert("RGB")
    pixel_values = processor(images=raw_image, return_tensors="pt").pixel_values.to(device)

    input_ids = processor(text=query, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)

    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, default="microsoft/git-base-textvqa")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    model.to(device)

    samples = json.load(open(args.infile, "r"))

    for sample in tqdm.tqdm(samples):
        q = sample["question"]
        image_file = os.path.join(args.img_dir, sample["image"])
        output = eval_model(processor, model, image_file, q)

        output = output.strip().replace(".", '').lower()
        sample["output"] = output
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    json.dump(samples, open(args.outfile, "w"), indent=4)
