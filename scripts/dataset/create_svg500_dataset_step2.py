"""
Pull from Icey444/svg500, reformat to VG format, and push to Icey444/svg500_in_vg

This pre-converts the SVG format to VG format so that the dataset is ready
to use directly with SceneGraphData without runtime conversion.
"""

import argparse
from datasets import Dataset, load_dataset, Features, Value, Sequence, Image
from tqdm import tqdm
import os
import sys
import json

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def convert_svg_to_vg_format(case):
    """
    Convert SVG dataset format to VG format for SceneGraphData compatibility.
    Ensures minimum width/height of 1 pixel.
    """
    regions = case.get('regions', [])
    scene_graph = case.get('scene_graph', {})
    
    # Parse scene_graph if it's a JSON string
    if isinstance(scene_graph, str):
        try:
            scene_graph = json.loads(scene_graph)
        except json.JSONDecodeError:
            scene_graph = {}
    
    objects = scene_graph.get('objects', [])
    relations = scene_graph.get('relations', [])
    
    # Build VG-style objects dict
    vg_objects = {}
    for idx, region in enumerate(regions):
        bbox = region.get('bbox')
        if bbox is None:
            raise ValueError(
                f"Missing bbox for region {idx} (object: {region.get('object', 'unknown')}). "
                f"All regions must have bbox coordinates."
            )
        
        # Get object name from scene_graph if available
        obj_name = objects[idx] if idx < len(objects) else region.get('object', 'unknown')
        
        # SVG bbox format is [x, y, w, h] in pixel coordinates (not normalized)
        if len(bbox) == 4:
            x, y, w, h = bbox
            # Convert to integers and ensure minimum width/height of 1
            x = int(x)
            y = int(y)
            w = max(1, int(w))
            h = max(1, int(h))
        else:
            raise ValueError(
                f"Invalid bbox for region {idx} (object: {obj_name}). "
                f"Expected 4 coordinates [x, y, w, h], got: {bbox}."
            )
        
        vg_objects[str(idx)] = {
            'object_id': idx,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'names': [obj_name],
            'synsets': [],
            'attributes': []
        }
    
    # Build VG-style relationships list
    vg_relationships = []
    for rel in relations:
        if len(rel) == 3:
            subj_idx, obj_idx, predicate = rel
            
            if subj_idx < len(regions) and obj_idx < len(regions):
                subj_name = objects[subj_idx] if subj_idx < len(objects) else f"object_{subj_idx}"
                obj_name = objects[obj_idx] if obj_idx < len(objects) else f"object_{obj_idx}"
                
                vg_relationships.append({
                    'predicate': predicate,
                    'synsets': [],
                    'subject': vg_objects[str(subj_idx)],
                    'object': vg_objects[str(obj_idx)]
                })
    
    # Create VG-format case
    vg_case = case.copy()
    vg_case['sg'] = {
        'objects': vg_objects,
        'relationships': vg_relationships
    }
    # Keep original image dimensions (SVG uses actual pixel sizes)
    # Width/height are already set in the original case if available
    if 'width' not in vg_case:
        vg_case['width'] = 640  # Default from COCO/ADE20K typical sizes
    if 'height' not in vg_case:
        vg_case['height'] = 480
    
    return vg_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset', type=str, default="Icey444/svg500",
                        help='Input HuggingFace dataset path')
    parser.add_argument('--output_dataset', type=str, default="Icey444/svg500_in_vg",
                        help='Output HuggingFace dataset path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_shards', type=int, default=16,
                        help='Number of shards for parallel upload')
    parser.add_argument('--private', action='store_true',
                        help='Make the dataset private on HuggingFace')
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.input_dataset}...")
    dataset = load_dataset(args.input_dataset, split='train')
    total_samples = len(dataset)
    print(f"Total samples in dataset: {total_samples}")
    
    # Convert each sample to VG format
    sampled_data = []
    print("Converting to VG format...")
    
    for idx in tqdm(range(total_samples)):
        sample = dict(dataset[idx])
        
        try:
            # Convert to VG format
            vg_sample = convert_svg_to_vg_format(sample)
            
            # Serialize sg dict to JSON string for HuggingFace
            if 'sg' in vg_sample and isinstance(vg_sample['sg'], dict):
                vg_sample['sg'] = json.dumps(vg_sample['sg'])
            
            sampled_data.append(vg_sample)
        except Exception as e:
            print(f"\nWarning: Failed to process sample {idx} (image_id: {sample.get('image_id', 'unknown')}): {e}")
    
    print(f"\nSuccessfully processed {len(sampled_data)} samples in VG format")
    
    if len(sampled_data) == 0:
        print("ERROR: No samples successfully converted!")
        return
    
    # Create HuggingFace dataset with VG-compatible schema
    print("\nCreating HuggingFace dataset with VG schema...")
    
    # First check what fields are in the first sample
    print(f"Sample keys: {sampled_data[0].keys()}")
    
    # Create dataset without explicit features to let it infer
    print("\nCreating Arrow dataset (this may take a few minutes with images)...")
    import sys
    sys.stdout.flush()
    dataset = Dataset.from_list(sampled_data)
    print("✓ Dataset created successfully")
    
    print(f"\nDataset info:")
    print(f"  Num samples: {len(dataset)}")
    print(f"  Features: {dataset.features}")
    
    # Push to HuggingFace
    print(f"\nPushing to HuggingFace: {args.output_dataset}")
    print(f"Private: {args.private}")
    print(f"Number of shards: {args.num_shards}")
    
    dataset.push_to_hub(
        args.output_dataset,
        private=args.private,
        num_shards=args.num_shards
    )
    
    print(f"\n✓ Successfully pushed {len(dataset)} samples to {args.output_dataset}")
    print(f"  Dataset URL: https://huggingface.co/datasets/{args.output_dataset}")
    print(f"\nThis dataset has 'sg' field in VG format and can be used directly with SceneGraphData.")


if __name__ == "__main__":
    main()