"""
Create a random sample of 500 items from the SVG dataset with images loaded
and push to HuggingFace as Icey444/svg500

Follows the existing design in utils/utils.py for loading SVG images from local directory.
"""

import argparse
import random
from datasets import Dataset, Image, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm
import os
import sys
import json

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.svg import load_svg
from utils.utils import load_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Number of samples to randomly select')
    parser.add_argument('--output_dataset', type=str, default="Icey444/svg500",
                        help='Output HuggingFace dataset path')
    parser.add_argument('--svg_image_dir', type=str, default='data/svg_images/all',
                        help='Directory containing SVG images')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--private', action='store_true',
                        help='Make the dataset private on HuggingFace')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print(f"Loading all SVG samples from cache/HuggingFace...")
    all_samples = load_svg(num_samples=None)  # Load all samples
    total_samples = len(all_samples)
    print(f"Total samples in original dataset: {total_samples}")
    
    # Load original HF dataset to get the exact schema/features
    print("Loading original HuggingFace dataset schema...")
    original_hf_dataset = load_dataset("jamepark3922/svg", split='train')
    original_features = original_hf_dataset.features
    print(f"Original features: {original_features}")
    
    # Random sampling
    if args.num_samples >= total_samples:
        print(f"Warning: Requested {args.num_samples} samples but dataset only has {total_samples}")
        selected_indices = list(range(total_samples))
    else:
        selected_indices = random.sample(range(total_samples), args.num_samples)
        selected_indices.sort()  # Sort for reproducibility
    
    print(f"Randomly selected {len(selected_indices)} samples")
    
    # Convert to absolute path
    svg_image_dir = os.path.abspath(args.svg_image_dir)
    print(f"Looking for images in: {svg_image_dir}")
    
    # Prepare data with images loaded (following utils/utils.py design)
    sampled_data = []
    print("Loading images for selected samples...")
    
    for idx in tqdm(selected_indices):
        sample = all_samples[idx].copy()
        
        # Load image - images are directly under svg_image_dir, not in subdirectories
        if 'image_id' in sample:
            image_id = sample['image_id']
            # Extract just the filename if image_id contains path separators
            if '/' in image_id:
                image_filename = os.path.basename(image_id)
            else:
                image_filename = image_id
            
            image_path = os.path.join(svg_image_dir, image_filename)
            
            # Handle symlinks: resolve relative symlink targets from workspace root
            if os.path.islink(image_path):
                target = os.readlink(image_path)
                # If target is relative, resolve it from workspace root
                if not os.path.isabs(target):
                    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(svg_image_dir)))
                    # Ensure scene_graph is JSON string (matching HF format)
                    if 'scene_graph' in sample and isinstance(sample['scene_graph'], dict):
                        sample['scene_graph'] = json.dumps(sample['scene_graph'])
                    image_path = os.path.join(workspace_root, target)
            
            # Check if file exists and load it
            if os.path.exists(image_path):
                try:
                    sample['image'] = load_image(image_path)
                    sampled_data.append(sample)
                except Exception as e:
                    print(f"Warning: Failed to load image at {image_path}: {e}")
            else:
                print(f"Warning: Image not found for sample {idx}: {image_path}")
        else:
            print(f"Warning: No image_id found for sample {idx}")
    
    print(f"\nSuccessfully loaded {len(sampled_data)} samples with images")
    
    # Use the original features schema and add image column
    features = original_features.copy()
    features['image'] = Image()
    
    dataset = Dataset.from_list(sampled_data, features=featuresh or python scripts/download_svg_images.py")
        return
    
    # Create HuggingFace dataset
    print("\nCreating HuggingFace dataset...")
    dataset = Dataset.from_list(sampled_data)
    
    # Cast image column to Image type
    if 'image' in dataset.column_names:
        dataset = dataset.cast_column('image', Image())
    
    print(f"Dataset info:")
    print(f"  Num samples: {len(dataset)}")
    print(f"  Features: {dataset.features}")
    
    # Push to HuggingFace
    print(f"\nPushing to HuggingFace: {args.output_dataset}")
    print(f"Private: {args.private}")
    
    dataset.push_to_hub(
        args.output_dataset,
        private=args.private
    )
    
    print(f"\n✓ Successfully pushed {len(dataset)} samples to {args.output_dataset}")
    print(f"  Dataset URL: https://huggingface.co/datasets/{args.output_dataset}")


if __name__ == "__main__":
    main()
