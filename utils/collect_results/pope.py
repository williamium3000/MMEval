#!/usr/bin/env python3
"""
Collect POPE/verify results from *_with_both_answers.json files.
Reports count of yes/no QAs and two accuracies (dynamic = response→Qwen, single = VLM)
both as question-level (micro) and image-level (macro) averages.
"""

import json
import csv
import os
import argparse
from collections import defaultdict


def find_with_both_answers_files(base_dir):
    """Find all *_with_both_answers.json files under base_dir."""
    eval_files = []
    base_dir = os.path.abspath(base_dir)
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith('_with_both_answers.json'):
                full_path = os.path.join(root, f)
                model_name = f.replace('_with_both_answers.json', '')
                eval_files.append((model_name, full_path))
    return sorted(eval_files, key=lambda x: x[0])


def extract_pope_metrics(json_path):
    """
    From a with_both_answers.json: count evaluated QAs and compute
    - Question-level (micro): accuracy over all QAs
    - Image-level (macro): per-image accuracy then average over images
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading {json_path}: {e}")
        return None

    # dynamic = response→Qwen (same as batch log "dynamic (response)"); single = image→VLM ("single (VLM)")
    dyn_ok_qa = 0
    dyn_ev_qa = 0
    single_ok_qa = 0
    single_ev_qa = 0
    # Per-image: (correct, total) for dynamic and single
    image_dyn = []
    image_single = []

    for sample in data:
        img_dyn_ok, img_dyn_ev = 0, 0
        img_single_ok, img_single_ev = 0, 0
        for turn in sample.get('conversations', []):
            for qa in turn.get('dsg_qa', []):
                # dynamic_is_correct = accuracy of dynamic_response (Qwen from transcript)
                if 'dynamic_is_correct' in qa:
                    dyn_ev_qa += 1
                    img_dyn_ev += 1
                    if qa['dynamic_is_correct']:
                        dyn_ok_qa += 1
                        img_dyn_ok += 1
                # single_is_correct = accuracy of single_response (VLM from image+questions)
                if 'single_is_correct' in qa:
                    single_ev_qa += 1
                    img_single_ev += 1
                    if qa['single_is_correct']:
                        single_ok_qa += 1
                        img_single_ok += 1
        if img_dyn_ev > 0:
            image_dyn.append((img_dyn_ok, img_dyn_ev))
        if img_single_ev > 0:
            image_single.append((img_single_ok, img_single_ev))

    # Question-level (micro): total correct / total count
    dynamic_acc_qa = (dyn_ok_qa / dyn_ev_qa * 100) if dyn_ev_qa else 0
    single_acc_qa = (single_ok_qa / single_ev_qa * 100) if single_ev_qa else 0
    n_qa = max(dyn_ev_qa, single_ev_qa)

    # Image-level (macro): mean of (per-image accuracy)
    if image_dyn:
        dynamic_acc_image = sum(ok / ev for ok, ev in image_dyn) / len(image_dyn) * 100
        n_images_dyn = len(image_dyn)
    else:
        dynamic_acc_image = 0
        n_images_dyn = 0
    if image_single:
        single_acc_image = sum(ok / ev for ok, ev in image_single) / len(image_single) * 100
        n_images_single = len(image_single)
    else:
        single_acc_image = 0
        n_images_single = 0
    n_images = max(n_images_dyn, n_images_single)

    return {
        'model': '',
        'n_qa': n_qa,
        'n_images': n_images,
        'dynamic_acc_qa': round(dynamic_acc_qa, 2),
        'single_acc_qa': round(single_acc_qa, 2),
        'dynamic_acc_image': round(dynamic_acc_image, 2),
        'single_acc_image': round(single_acc_image, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Collect POPE verify results from *_with_both_answers.json'
    )
    parser.add_argument('--input_dir', type=str,
                        default='work_dirs/vg/final_run_v18_gpt4o_completed',
                        help='Directory containing *_with_both_answers.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV (default: <input_dir>/pope_results_summary.csv)')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.input_dir, 'pope_results_summary.csv')
    elif os.path.dirname(args.output) == '':
        args.output = os.path.join(args.input_dir, args.output)

    print(f"📊 Collecting POPE (with_both_answers) results from: {args.input_dir}")
    print(f"💾 Output CSV: {args.output}\n")

    eval_files = find_with_both_answers_files(args.input_dir)
    if not eval_files:
        print(f"⚠️  No *_with_both_answers.json found under {args.input_dir}")
        return

    print(f"📁 Found {len(eval_files)} file(s)\n")

    all_metrics = []
    seen_models = {}
    for model_name, json_path in eval_files:
        print(f"   📄 {model_name}")
        m = extract_pope_metrics(json_path)
        if m is None:
            continue
        m['model'] = model_name
        seen_models[model_name] = m
    all_metrics = list(seen_models.values())

    if not all_metrics:
        print("⚠️  No valid metrics extracted")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = [
        'model', 'n_qa', 'n_images',
        'dynamic_acc_qa', 'single_acc_qa',
        'dynamic_acc_image', 'single_acc_image',
    ]
    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_metrics)

    print(f"\n✅ Results written to: {args.output}")
    print(f"📊 Total models: {len(all_metrics)}\n")

    # Table: question-level (micro) — same order as batch log: dynamic (response) first, single (VLM) second
    print("=" * 100)
    print("POPE Verify — Question-level (micro: avg over all yes/no QAs)")
    print("  dynamic = response→Qwen (batch log 'dynamic (response)'); single = image→VLM (batch log 'single (VLM)')")
    print("=" * 100)
    print(f"{'Model':<42} {'n_qa':<8} {'dynamic (response→Qwen)':<24} {'single (image→VLM)':<20}")
    print("-" * 100)
    for m in sorted(all_metrics, key=lambda x: -x['dynamic_acc_qa']):
        print(f"{m['model']:<42} {m['n_qa']:<8} {m['dynamic_acc_qa']:<24.2f} {m['single_acc_qa']:<20.2f}")
    print("=" * 100)

    # Table: image-level (macro)
    print("\nPOPE Verify — Image-level (macro: avg of per-image accuracy)")
    print("  dynamic = response→Qwen; single = image→VLM")
    print("=" * 100)
    print(f"{'Model':<42} {'n_images':<10} {'dynamic (response→Qwen)':<24} {'single (image→VLM)':<20}")
    print("-" * 100)
    for m in sorted(all_metrics, key=lambda x: -x['dynamic_acc_image']):
        print(f"{m['model']:<42} {m['n_images']:<10} {m['dynamic_acc_image']:<24.2f} {m['single_acc_image']:<20.2f}")
    print("=" * 100)
    print(f"\n💾 Full results: {args.output}\n")


if __name__ == '__main__':
    main()
