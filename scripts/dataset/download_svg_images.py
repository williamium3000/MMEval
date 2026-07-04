#!/usr/bin/env python3
"""
Download images for SVG dataset from ADE20K, COCO, and Visual Genome.

This script automates downloading images needed for SVG evaluation.
Images are organized by source dataset in subdirectories.

Usage:
    python scripts/download_svg_images.py
    
Note: Use scripts/prepare_svg.sh to download annotations, images, and create symlinks.
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import tarfile


def download_file(url, output_path, desc="Downloading"):
    """Download a file with progress bar."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    return output_path


def extract_archive(archive_path, extract_to):
    """Extract zip or tar archive."""
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {archive_path.name}...")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    
    print(f"✓ Extracted to {extract_to}")


def download_ade20k(output_dir):
    """Download ADE20K images from HuggingFace dataset."""
    print("\n" + "="*60)
    print("Downloading ADE20K Images from HuggingFace")
    print("="*60)
    
    ade20k_dir = Path(output_dir) / "ade20k"
    ade20k_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
        from PIL import Image
        import io
        
        print(f"Loading ADE20K dataset from HuggingFace...")
        print(f"Destination: {ade20k_dir}")
        
        # Load the dataset
        ds = load_dataset("1aurent/ADE20K", split="train", streaming=False)
        
        print(f"Downloading {len(ds)} images...")
        
        # Save each image with its filename
        for idx, sample in enumerate(tqdm(ds, desc="Saving ADE20K images")):
            filename = sample['filename']
            image = sample['image']  # PIL Image
            
            # Save the image
            image_path = ade20k_dir / filename
            image.save(image_path)
            
            if (idx + 1) % 1000 == 0:
                print(f"  Saved {idx + 1}/{len(ds)} images...")
        
        print(f"✓ ADE20K images downloaded successfully to {ade20k_dir}")
        return True
    except Exception as e:
        print(f"✗ Error downloading ADE20K: {e}")
        print(f"Make sure you have 'datasets' installed: pip install datasets")
        return False


def download_coco(output_dir):
    """Download COCO train2017 images."""
    print("\n" + "="*60)
    print(f"Downloading COCO 2017 train Images")
    print("="*60)
    
    coco_dir = Path(output_dir) / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO images URL
    url = "http://images.cocodataset.org/zips/train2017.zip"
    
    print(f"Source: {url}")
    print(f"Destination: {coco_dir}")
    
    # Download
    zip_path = coco_dir / "train2017.zip"
    try:
        download_file(url, zip_path, desc="Downloading COCO train2017")
        extract_archive(zip_path, coco_dir)
        
        # Clean up zip file
        zip_path.unlink()
        
        print(f"✓ COCO 2017 train images downloaded successfully to {coco_dir}")
        return True
    except Exception as e:
        print(f"✗ Error downloading COCO: {e}")
        print(f"Note: You may need to download COCO images manually from https://cocodataset.org/")
        return False


def download_visual_genome(output_dir, part=1):
    """Download Visual Genome images."""
    print("\n" + "="*60)
    print(f"Downloading Visual Genome Images (Part {part})")
    print("="*60)
    
    vg_dir = Path(output_dir) / "visual_genome"
    vg_dir.mkdir(parents=True, exist_ok=True)
    
    # Visual Genome images URL
    url = f"https://cs.stanford.edu/people/rak248/VG_100K_2/images{part if part == 2 else ''}.zip"
    
    print(f"Source: {url}")
    print(f"Destination: {vg_dir}")
    
    # Download
    zip_path = vg_dir / f"images_part{part}.zip"
    try:
        download_file(url, zip_path, desc=f"Downloading VG Part {part}")
        extract_archive(zip_path, vg_dir)
        
        # Clean up zip file
        zip_path.unlink()
        
        print(f"✓ Visual Genome Part {part} downloaded successfully to {vg_dir}")
        return True
    except Exception as e:
        print(f"✗ Error downloading Visual Genome Part {part}: {e}")
        print(f"Note: You may need to download VG images manually from https://homes.cs.washington.edu/~ranjay/visualgenome/api.html")
        return False


def main():
    output_dir = Path("data/svg_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SVG Image Download Script")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Datasets: ADE20K, COCO train2017, Visual Genome")
    print(f"{'='*60}\n")
    
    results = {}
    
    # Download all datasets
    results["ade20k"] = download_ade20k(output_dir)
    # results["coco"] = download_coco(output_dir)
    # results["vg_part1"] = download_visual_genome(output_dir, part=1)
    # results["vg_part2"] = download_visual_genome(output_dir, part=2)
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{dataset:20s}: {status}")
    
    print("\n" + "="*60)
    print("Images Downloaded")
    print("="*60)
    print(f"ADE20K: {output_dir / 'ade20k' / 'ADEChallengeData2016' / 'images' / 'training'}")
    print(f"COCO: {output_dir / 'coco' / 'train2017'}")
    print(f"VG Part 1: {output_dir / 'visual_genome' / 'VG_100K'}")
    print(f"VG Part 2: {output_dir / 'visual_genome' / 'VG_100K_2'}")
    print("\n" + "="*60)
    print("Next Step")
    print("="*60)
    print("Run: bash scripts/prepare_svg.sh")
    print("  This will create symlinks and complete the setup.")
    print("="*60 + "\n")
    
    # Exit with error if any download failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
