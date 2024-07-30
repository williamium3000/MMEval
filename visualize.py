from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import argparse
import json
import os

from utils.coco import format_case_coco
from matplotlib.gridspec import GridSpec

coco17_instance = COCO("data/coco/annotations/instances_val2017.json")
coco17_caption = COCO("data/coco/annotations/captions_val2017.json")
cats17 = coco17_instance.loadCats(coco17_instance.getCatIds())
id_name_mapping17 = {cat["id"]: cat["name"] for cat in cats17}
coco14_instance = COCO("data/coco/annotations/instances_val2014.json")
coco14_caption = COCO("data/coco/annotations/captions_val2014.json")
cats14 = coco14_instance.loadCats(coco14_instance.getCatIds())
id_name_mapping14 = {cat["id"]: cat["name"] for cat in cats14}

parser = argparse.ArgumentParser()
parser.add_argument('json', type=str)
parser.add_argument('outdir', type=str)
parser.add_argument('--anno', action="store_true")
args = parser.parse_args()


os.makedirs(args.outdir, exist_ok=True)
json_data = json.load(open(args.json))
for i, sample in enumerate(json_data):
    format_anno = format_case_coco(sample)
    img_id = sample["image_id"]
    img = coco17_instance.loadImgs([img_id])[0]
    # visual the result conversationq
    # first display the image with bbox and captions
    # then display the conversation in the image
    
    I = io.imread(os.path.join("data/coco/val2017", sample["file_name"]))
    # fig, ax = plt.subplots(1, 2)
    # ax.imshow(I)
    # plt.axis('off')
    fig = plt.figure(figsize=(72, 48))

    # Define the GridSpec
    gs = GridSpec(1, 2, width_ratios=[1, 1.5])  # 1:1.5 ratio between image and text

    # Create the subplot for the image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(I)
    ax1.axis('off')  # Hide the axis
    
    if args.anno:
        annIds = coco17_instance.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco17_instance.loadAnns(annIds)
        coco17_instance.showAnns(anns)
    
    
    conversation_text = format_anno + "\n\n" + "\n\n".join([f"Round {conv['round_id']}:\nPrompt: {conv['prompt']}\nResponse: {conv['response']}" for conv in sample["conversations"]])
    # plt.figtext(0, 0.5, conversation_text, wrap=True, horizontalalignment='left', verticalalignment='center', fontsize=12, bbox=dict(facecolor='none', edgecolor='black'))
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')  # Hide the axis
    ax2.text(0, 0.5, conversation_text, wrap=True, horizontalalignment='left', verticalalignment='center', fontsize=28)

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{i}.png"))