"""
Synthetic Visual Genome (SVG) Dataset Loader
Loads the SVG dataset in VG format from HuggingFace: Icey444/svg5000_in_vg

The dataset has been pre-converted to VG format with 'sg' key for direct use with SceneGraphData.
"""

from datasets import load_dataset
import tqdm
import json
from typing import List, Dict, Optional


def load_svg(num_samples: Optional[int] = None, split: str = "train_500_augmented") -> List[Dict]:
    """
    Load SVG dataset in VG format.

    Loads from Icey444/svg5000_in_vg which has pre-converted SVG data to VG format
    with 'sg' key compatible with SceneGraphData.

    Args:
        num_samples: Number of samples to load (None = all)
        split: HF split to load. "train" = original noun-phrase names with empty
            attributes; "train_500_augmented" (default) = first 500 rows with
            cleaned head-noun names and OpenAI-extracted attributes.

    Returns:
        List of sample dictionaries with 'sg' key in VG format, images, and metadata.
    """
    print(f"Loading SVG dataset in VG format from Icey444/svg5000_in_vg [{split}]...")
    dataset = load_dataset("Icey444/svg5000_in_vg", split=split)
    
    # Only load the requested number of samples
    num_to_load = num_samples if num_samples is not None else len(dataset)
    num_to_load = min(num_to_load, len(dataset))
    
    all_samples = []
    print(f"Processing {num_to_load} samples...")
    for idx in tqdm.tqdm(range(num_to_load)):
        sample = dict(dataset[idx])
        
        # Parse sg if it's a JSON string (deserialize from HuggingFace format)
        if 'sg' in sample and isinstance(sample['sg'], str):
            try:
                sample['sg'] = json.loads(sample['sg'])
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse sg for sample {idx}")
        
        # Parse scene_graph if it's a JSON string (for reference)
        if 'scene_graph' in sample and isinstance(sample['scene_graph'], str):
            try:
                sample['scene_graph'] = json.loads(sample['scene_graph'])
            except json.JSONDecodeError:
                pass
        
        all_samples.append(sample)
    
    print(f"Loaded {len(all_samples)} SVG samples in VG format")
    return all_samples


def format_case_svg(case: Dict, use_region: bool = False) -> str:
    """
    Format SVG sample in VG style for display/evaluation.
    
    Args:
        case: Sample dictionary from SVG (with 'sg' key in VG format)
        use_region: Whether to include region descriptions (not used for SVG)
    
    Returns:
        Formatted string representation matching VG format
    """
    formatted = "Instances:\n"
    H = case.get("height", 1000)
    W = case.get("width", 1000)
    
    sg = case.get("sg", {})
    
    # Format objects
    for ori_id, ins in sg.get("objects", {}).items():
        object_id = ins["object_id"]
        x, y, w, h = ins['x'], ins['y'], ins['w'], ins['h']
        x1, y1, x2, y2 = x / W, y / H, (x + w) / W, (y + h) / H
        
        if ins.get("attributes", []) is None or len(ins.get("attributes", [])) == 0:
            cur_attr = "none"
        else:
            attrs = ins.get("attributes", [])
            cur_attr = ", ".join(attrs)
        
        formatted += f"instance {object_id}, {ins['names'][0]}, bbox: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}), attributes: {cur_attr}\n"
    
    # Format relationships
    formatted += "\nRelation between the above instances:\n"
    for rel in sg.get("relationships", []):
        formatted += f"{rel['subject']['names'][0]} (instance {rel['subject']['object_id']}) {rel['predicate'].lower()} {rel['object']['names'][0]} (instance {rel['object']['object_id']})\n"
    
    return formatted


if __name__ == "__main__":
    print(format_case_svg(load_svg(num_samples=5)[0]))
