from .coco import load_coco2017
from .vg import load_vg
from .svg import load_svg

import os
from PIL import Image
import argparse
import torch
import json
import requests
from PIL import Image
from io import BytesIO
import re
import os
import tqdm

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_data(args):
    if args.dataset == "coco":
        samples =  load_coco2017(args.num_samples)
        for sample in samples:
            sample["image"] = load_image(os.path.join("data/coco/val2017", sample["file_name"]))
            
    elif args.dataset == "vg":
        samples = load_vg(args.num_samples)
        
    elif args.dataset == "svg":
        # Load from Icey444/svg500_in_vg - already has images and VG format
        print("Loading SVG dataset in VG format from HuggingFace (Icey444/svg500_in_vg)...")
        samples = load_svg(num_samples=args.num_samples)
        print(f"Loaded {len(samples)} SVG samples with images in VG format")
        
    else:
        raise ValueError("Unknown dataset")
    return samples
