#!/usr/bin/env python3
# utils/collect_results/chair.py

import json
import csv
import os
import argparse
from pathlib import Path
from collections import defaultdict

def find_all_eval_files(base_dir):
    """Find all CHAIR evaluation JSON files"""
    eval_files = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith('hallucinated_words_') and file.endswith('.json'):
                full_path = os.path.join(root, file)
                # Extract model name from filename: hallucinated_words_<model>.json
                model_name = file.replace('hallucinated_words_', '').replace('.json', '')
                eval_files.append((model_name, full_path))
    
    return sorted(eval_files)

def extract_metrics(json_path):
    """Extract metrics from CHAIR evaluation JSON"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        overall = data.get('overall_metrics', {})
        by_qtype = data.get('by_qtype', {})
        
        metrics = {
            'model': '',
            # Overall metrics
            'overall_CHAIRs': overall.get('CHAIRs', 0),
            'overall_CHAIRi': overall.get('CHAIRi', 0),
            'overall_CHAIRi_v2': overall.get('CHAIRi_v2', 0),
            'overall_Coverage_avg': overall.get('Coverage_avg', 0),
            'overall_Coverage_all': overall.get('Coverage_all', 0),
        }
        
        # Per question type metrics
        for q_type in ['regular', 'follow-up', 'adversarial', 'unanswerable']:
            type_data = by_qtype.get(q_type, {})
            prefix = q_type.replace('-', '_')
            
            metrics[f'{prefix}_CHAIRs'] = type_data.get('CHAIRs', 0)
            metrics[f'{prefix}_CHAIRi'] = type_data.get('CHAIRi', 0)
            metrics[f'{prefix}_CHAIRi_v2'] = type_data.get('CHAIRi_v2', 0)
            metrics[f'{prefix}_Coverage_avg'] = type_data.get('Coverage_avg', 0)
            metrics[f'{prefix}_Coverage_all'] = type_data.get('Coverage_all', 0)
            metrics[f'{prefix}_num_caps'] = type_data.get('num_caps', 0)
            metrics[f'{prefix}_num_hallucinated_caps'] = type_data.get('num_hallucinated_caps', 0)
        
        return metrics
    except Exception as e:
        print(f"❌ Error reading {json_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Collect CHAIR results into CSV')
    parser.add_argument('--input_dir', type=str, 
                       default='work_dirs/vg/final_run_v18_gpt4o_completed',
                       help='Directory containing evaluation results')
    parser.add_argument('--output', type=str,
                       default=None,
                       help='Output CSV file (default: <input_dir>/chair_results_summary.csv)')
    
    args = parser.parse_args()
    
    # Set default output to input directory if not specified
    if args.output is None:
        args.output = os.path.join(args.input_dir, 'chair_results_summary.csv')
    else:
        # If output is specified but is just a filename, place it in input_dir
        if os.path.dirname(args.output) == '':
            args.output = os.path.join(args.input_dir, args.output)
    
    print(f"📊 Collecting CHAIR results from: {args.input_dir}")
    print(f"💾 Output CSV: {args.output}")
    print()
    
    # Find all evaluation files
    eval_files = find_all_eval_files(args.input_dir)
    
    if not eval_files:
        print(f"⚠️  No evaluation files found in {args.input_dir}")
        print(f"   Looking for files matching: hallucinated_words_*.json")
        return
    
    print(f"📁 Found {len(eval_files)} evaluation files")
    
    # Extract metrics from all files
    all_metrics = []
    seen_models = {}  # Track models to deduplicate (keep last occurrence)
    
    for model_name, json_path in eval_files:
        print(f"   📄 Processing: {model_name}")
        metrics = extract_metrics(json_path)
        
        if metrics:
            metrics['model'] = model_name
            # Store in dict to deduplicate (last one wins)
            seen_models[model_name] = metrics
    
    # Convert to list and deduplicate
    all_metrics = list(seen_models.values())
    
    if not all_metrics:
        print("⚠️  No valid metrics extracted")
        return
    
    # Report deduplication
    if len(eval_files) > len(all_metrics):
        print(f"   ℹ️  Deduplicated: {len(eval_files)} files → {len(all_metrics)} unique models")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Write to CSV
    fieldnames = [
        'model',
        # Overall
        'overall_CHAIRs',
        'overall_CHAIRi',
        'overall_CHAIRi_v2',
        'overall_Coverage_avg',
        'overall_Coverage_all',
        # Regular
        'regular_CHAIRs',
        'regular_CHAIRi',
        'regular_CHAIRi_v2',
        'regular_Coverage_avg',
        'regular_Coverage_all',
        'regular_num_caps',
        'regular_num_hallucinated_caps',
        # Follow-up
        'follow_up_CHAIRs',
        'follow_up_CHAIRi',
        'follow_up_CHAIRi_v2',
        'follow_up_Coverage_avg',
        'follow_up_Coverage_all',
        'follow_up_num_caps',
        'follow_up_num_hallucinated_caps',
        # Adversarial
        'adversarial_CHAIRs',
        'adversarial_CHAIRi',
        'adversarial_CHAIRi_v2',
        'adversarial_Coverage_avg',
        'adversarial_Coverage_all',
        'adversarial_num_caps',
        'adversarial_num_hallucinated_caps',
        # Unanswerable
        'unanswerable_CHAIRs',
        'unanswerable_CHAIRi',
        'unanswerable_CHAIRi_v2',
        'unanswerable_Coverage_avg',
        'unanswerable_Coverage_all',
        'unanswerable_num_caps',
        'unanswerable_num_hallucinated_caps',
    ]
    
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for metrics in all_metrics:
            writer.writerow(metrics)
    
    print(f"\n✅ Results written to: {args.output}")
    print(f"📊 Total models: {len(all_metrics)}")
    
    # Print summary table
    print("\n" + "="*120)
    print("SUMMARY TABLE (Top 10 by Overall CHAIRs - lower is better)")
    print("="*120)
    
    # Sort by CHAIRs (lower is better for hallucination rate)
    sorted_metrics = sorted(all_metrics, key=lambda x: x['overall_CHAIRs'])
    
    print(f"{'Rank':<6} {'Model':<40} {'CHAIRs':<10} {'CHAIRi':<10} {'CHAIRi_v2':<12} {'Coverage-avg':<12} {'Coverage-all':<12}")
    print("-"*120)
    
    for rank, metrics in enumerate(sorted_metrics[:10], 1):
        print(f"{rank:<6} {metrics['model']:<40} "
              f"{metrics['overall_CHAIRs']:<10.2f} "
              f"{metrics['overall_CHAIRi']:<10.2f} "
              f"{metrics['overall_CHAIRi_v2']:<12.2f} "
              f"{metrics['overall_Coverage_avg']:<12.2f} "
              f"{metrics['overall_Coverage_all']:<12.2f}")
    
    print("="*120)
    print(f"\n💾 Full results saved to: {args.output}\n")

if __name__ == '__main__':
    main()
