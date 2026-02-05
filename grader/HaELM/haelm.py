import argparse
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import json


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def load_model(args, tokenizer):
    model = LlamaForCausalLM.from_pretrained(
        args.llama_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        model,
        args.checkpoint_path,
        force_download=True,
        torch_dtype=torch.float16,
    )

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--caption', type=str, default='data/vg.json')
    parser.add_argument('--conv', type=str, default='output/vg/icl.json')
    parser.add_argument("--llama_path", type=str, default="checkpoints/llama-7b-hf")
    parser.add_argument("--checkpoint_path", type=str, default="grader/HaELM/checkpoint")
    parser.add_argument('--outfile', type=str, default='output/vg/icl_haelm.json')
    parser.add_argument('--sample_num', type=int, default=100)

    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.llama_path)
    model = load_model(args, tokenizer)

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
    )

    caption_index = 0
    # caption_info = json.load(open(args.caption, 'r'))
    # caption_dict = {}
    # for item in caption_info:
    #     img_id = item["image_id"]
    #     if img_id not in caption_dict:
    #         caption_dict[img_id] = {
    #             "image_id": img_id,
    #             "captions": []
    #         }
    #     caption_dict[img_id]["captions"].append(item["caption"].strip())

    conv_data = json.load(open(args.conv, 'r'))

    sample_result = []
    sentence_result = []
    for sample in conv_data[:args.sample_num]:
        result_list = []
        img_id = f'vg_{sample["image_id"]}'
        for conv in sample["conversations"]:
            # if img_id in caption_dict.keys():
            #     captions = caption_dict[img_id]["captions"]
            # else:
            captions = [region["phrase"] for region in sample["sg"]["regions"]]

            prompt_format = "reference captions:\n{ref}.\nour caption:\n{response}\nIs our caption accurate?\n"
            caption_str = '. '.join(captions)

            conv["accurate"] = []
            response_list = conv["response"].strip().replace('\n', '').split('.')
            for respond_sentence in response_list:
                if len(respond_sentence) > 0:
                    prompt = prompt_format.format(ref=caption_str, response=respond_sentence)
                    inputs = tokenizer(prompt, return_tensors="pt")
                    input_ids = inputs["input_ids"].to(device)
                    with torch.no_grad():
                        generation_output = model.generate(
                            input_ids=input_ids,
                            generation_config=generation_config,
                            return_dict_in_generate=True,
                            output_scores=True,
                            max_new_tokens=1,
                        )

                    sentence = generation_output.sequences
                    sentence = tokenizer.decode(sentence.tolist()[0], skip_special_tokens=True)
                    result = sentence.split("\n")[-1].lower()

                    conv["accurate"].append(result)
                    result_list.append(result)
                    sentence_result.append(result)

        if 'no' in result_list:
            sample_result.append('no')
        else:
            sample_result.append('yes')

    with open(args.outfile, "w") as f:
        json.dump(conv_data, f, indent=4)

    print('Sentence level hallucination rate:', sentence_result.count('no') / len(sentence_result))
    print('Sample level hallucination rate:', sample_result.count('no') / len(sample_result))