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
import os
from dotenv import load_dotenv

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
    return completion.choices[0].message




def load_coco2017(img_id):
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
    