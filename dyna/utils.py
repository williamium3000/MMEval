from pycocotools.coco import COCO

coco17_instance = COCO("data/coco/annotations/instances_val2017.json")
coco17_caption = COCO("data/coco/annotations/captions_val2017.json")
cats17 = coco17_instance.loadCats(coco17_instance.getCatIds())
id_name_mapping17 = {cat["id"]: cat["name"] for cat in cats17}
coco14_instance = COCO("data/coco/annotations/instances_val2014.json")
coco14_caption = COCO("data/coco/annotations/captions_val2014.json")
cats14 = coco14_instance.loadCats(coco14_instance.getCatIds())
id_name_mapping14 = {cat["id"]: cat["name"] for cat in cats14}

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")

import tqdm
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




def load_sample_coco2017(img_id):
    img = coco17_instance.loadImgs(img_id)[0]
    
    annIds = coco17_instance.getAnnIds(imgIds=img_id, iscrowd=None)
    instance_anns = coco17_instance.loadAnns(coco17_instance.getAnnIds(imgIds=img_id, iscrowd=None))
    caption_anns = coco17_caption.loadAnns(coco17_caption.getAnnIds(imgIds=img['id']))
    
    return {
        "file_name": img["file_name"],
        "instances": [
                {"category": id_name_mapping17[instance["category_id"]], "bbox": instance["bbox"], "pixel_area": instance["area"]} for instance in instance_anns
            ],
        "captions": [
                caption["caption"] for caption in caption_anns
            ]
    }
    
def load_coco2017(debug=False):
    all_img_ids = coco17_instance.getImgIds()
    if debug:
        all_img_ids = all_img_ids[:30]
        
    samples = []
    print(f"loading coco2017: total {len(all_img_ids)}")
    for img_id in tqdm.tqdm(all_img_ids):
        case = load_sample_coco2017(img_id)
        samples.append(case)
    return samples
        

if __name__ == "__main__":
    load_coco2017()

    