#!/usr/bin/env python3
# scripts/graders/collect_mmhal_results.py

# python /raid/william/project/context-eval-mllm/utils/collect_results/mmhal.py --input_dir /raid/william/project/context-eval-mllm/work_dirs/vg/caption

import json
import csv
import os
import argparse
from pathlib import Path
from collections import defaultdict

def find_all_eval_files(base_dir):
    """Find all mmhal evaluation JSON files"""
    eval_files = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith('mmhal_') and file.endswith('.json'):
                full_path = os.path.join(root, file)
                model_name = file.replace('mmhal_', '').replace('.json', '')
                eval_files.append((model_name, full_path))
    
    return sorted(eval_files)

def write_summary_csv(rows, output_csv_path):
    """Write mmhal summary CSV to output path."""
    fieldnames = [
        'model',
        # Overall
        'overall_avg_score',
        'overall_hallucination_rate',
        'overall_total',
        # Unanswerable
        'unanswerable_avg_score',
        'unanswerable_hallucination_rate',
        'unanswerable_count',
        # Regular
        'regular_avg_score',
        'regular_hallucination_rate',
        'regular_count',
        # Adversarial
        'adversarial_avg_score',
        'adversarial_hallucination_rate',
        'adversarial_count',
        # Follow-up
        'follow_up_avg_score',
        'follow_up_hallucination_rate',
        'follow_up_count',
    ]

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in rows:
            writer.writerow(metrics)

def extract_metrics(json_path):
    """Extract metrics from evaluation JSON"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        overall = data.get('overall_metrics', {})
        by_type = data.get('metrics_by_q_type', {})
        
        metrics = {
            'model': '',
            # Overall metrics
            'overall_avg_score': overall.get('avg_score', 0),
            'overall_hallucination_rate': overall.get('hallucination_rate', 0),
            'overall_total': overall.get('total_evaluations', 0),
        }
        
        # Per question type metrics
        for q_type in ['unanswerable', 'regular', 'adversarial', 'follow-up']:
            type_data = by_type.get(q_type, {})
            prefix = q_type.replace('-', '_')
            
            metrics[f'{prefix}_avg_score'] = type_data.get('avg_score', 0)
            metrics[f'{prefix}_hallucination_rate'] = type_data.get('hallucination_rate', 0)
            metrics[f'{prefix}_count'] = type_data.get('count', 0)
        
        return metrics
    except Exception as e:
        print(f"❌ Error reading {json_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Collect MMHAL results into CSV')
    parser.add_argument('--input_dir', type=str, 
                       default='work_dirs/vg/final_run_v18_gpt4o_completed',
                       help='Directory containing evaluation results')
    parser.add_argument('--output', type=str,
                       default=None,
                       help='Output CSV file (default: <input_dir>/mmhal_results_summary.csv)')
    
    args = parser.parse_args()
    
    # Set default output to be under input directory
    if args.output is None:
        args.output = os.path.join(args.input_dir, 'mmhal_results_summary.csv')
    else:
        # If output is specified but is just a filename, place it in input_dir
        if os.path.dirname(args.output) == '':
            args.output = os.path.join(args.input_dir, args.output)
    
    print(f"📊 Collecting MMHAL results from: {args.input_dir}")
    print(f"💾 Output CSV: {args.output}")
    print()
    
    # Find all evaluation files
    eval_files = find_all_eval_files(args.input_dir)
    
    if not eval_files:
        print(f"⚠️  No evaluation files found in {args.input_dir}")
        return
    
    print(f"📁 Found {len(eval_files)} evaluation files")
    
    # Extract metrics from all files
    seen_models = {}  # Deduplicate by model (last one wins)
    
    for model_name, json_path in eval_files:
        print(f"   📄 Processing: {model_name}")
        metrics = extract_metrics(json_path)
        
        if metrics:
            metrics['model'] = model_name
            seen_models[model_name] = metrics
    
    all_metrics = list(seen_models.values())
    if len(eval_files) > len(all_metrics):
        print(f"   ℹ️  Deduplicated: {len(eval_files)} files → {len(all_metrics)} unique models")

    if not all_metrics:
        print("⚠️  No valid metrics extracted")
        return
    
    # Write global summary CSV
    write_summary_csv(all_metrics, args.output)
    
    print(f"\n✅ Results written to: {args.output}")
    print(f"📊 Total models: {len(all_metrics)}")

    # Also write a summary CSV inside each first-level subdir, named with subdir
    # Example: <input_dir>/<subdir>/mmhal_results_summary_<subdir>.csv
    if os.path.isdir(args.input_dir):
        try:
            first_level_subdirs = [
                d for d in os.listdir(args.input_dir)
                if os.path.isdir(os.path.join(args.input_dir, d))
            ]
        except Exception:
            first_level_subdirs = []

        wrote_any = False
        for subdir in sorted(first_level_subdirs):
            subdir_path = os.path.join(args.input_dir, subdir)
            sub_eval_files = find_all_eval_files(subdir_path)
            if not sub_eval_files:
                continue

            sub_seen = {}
            for model_name, json_path in sub_eval_files:
                m = extract_metrics(json_path)
                if m:
                    m['model'] = model_name
                    sub_seen[model_name] = m
            sub_rows = list(sub_seen.values())
            if not sub_rows:
                continue

            per_dir_csv = os.path.join(subdir_path, f"mmhal_results_summary_{subdir}.csv")
            write_summary_csv(sub_rows, per_dir_csv)
            wrote_any = True
            print(f"   🧾 Wrote per-subdir summary: {per_dir_csv} ({len(sub_rows)} model(s))")
        if wrote_any:
            print("   ✅ Per-subdir summaries complete")
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE (Top 10 by Overall Score)")
    print("="*100)
    
    # Sort by overall score
    sorted_metrics = sorted(all_metrics, key=lambda x: x['overall_avg_score'], reverse=True)
    
    print(f"{'Rank':<6} {'Model':<40} {'Avg Score':<12} {'Hallu Rate':<12} {'Total':<8}")
    print("-"*100)
    
    for rank, metrics in enumerate(sorted_metrics[:10], 1):
        print(f"{rank:<6} {metrics['model']:<40} "
              f"{metrics['overall_avg_score']:<12.4f} "
              f"{metrics['overall_hallucination_rate']:<12.4f} "
              f"{metrics['overall_total']:<8}")
    
    print("="*100)
    print(f"\n💾 Full results saved to: {args.output}\n")

if __name__ == '__main__':
    main()