# python /raid/william/project/context-eval-mllm/utils/check_output.py /raid/william/project/context-eval-mllm/work_dirs/vg/final_run_v18_gpt4o --auto-csv --copy-completed --clean-duplicates
# python /raid/william/project/context-eval-mllm/utils/check_output.py /raid/william/project/context-eval-mllm/work_dirs/vg/final_run_v17_gpt5 --auto-csv --copy-completed --clean-duplicates
# python /raid/william/project/context-eval-mllm/utils/check_output.py /raid/william/project/context-eval-mllm/work_dirs/vg/final_run_v18_gpt4o_conv --auto-csv --copy-completed --clean-duplicates
# python /raid/william/project/context-eval-mllm/utils/check_output.py /raid/william/project/context-eval-mllm/work_dirs/vg/ablation_v18_nocontext --auto-csv --copy-completed --clean-duplicates
# python /raid/william/project/context-eval-mllm/utils/check_output.py /raid/william/project/context-eval-mllm/work_dirs/vg/ablation_nonodeselect --auto-csv --copy-completed --clean-duplicates
# python /raid/william/project/context-eval-mllm/utils/check_output.py /raid/william/project/context-eval-mllm/work_dirs/vg/ablation_1round --auto-csv --copy-completed --clean-duplicates
# python /raid/william/project/context-eval-mllm/utils/check_output.py /raid/william/project/context-eval-mllm/work_dirs/vg/ablation_baseline --auto-csv --copy-completed --clean-duplicates
# python /raid/william/project/context-eval-mllm/utils/check_output.py /raid/william/project/context-eval-mllm/work_dirs/vg/final_run_v18_gpt4o_resume5 --auto-csv --copy-completed --clean-duplicates

#!/usr/bin/env python3
"""
Check progress of conversation generation in a work_dirs output directory.
Reports how many samples have 0, 1, or 2 goals completed.
"""

import os
import json
import argparse
import csv
import shutil
from collections import defaultdict


def extract_model_name(filename):
    """Extract model name from filename, handling _cache.json files."""
    # Remove .json extension
    name = filename.replace('.json', '')
    # Remove _cache suffix if present
    if name.endswith('_cache'):
        name = name[:-6]  # Remove '_cache'
    return name


def is_completed(goal_counts, is_nocontext=False, expected_goals=2, threshold=95):
    """Check if completion percentage is >= threshold (default 95%).
    
    Args:
        goal_counts: Dictionary with goal counts
        is_nocontext: If True, checks 1_goal count
        expected_goals: Number of goals expected per sample
        threshold: Completion threshold percentage (default 95)
    
    Returns:
        tuple: (is_completed_bool, completion_percentage)
    """
    total_samples = 100
    if is_nocontext:
        completed = goal_counts.get(1, 0)
        completion_pct = (completed / total_samples) * 100
    else:
        completed = goal_counts.get(expected_goals, 0)
        completion_pct = (completed / total_samples) * 100
    
    return completion_pct >= threshold, completion_pct


def check_progress(json_file, is_nocontext=False, expected_goals=2):
    """Check progress for a single JSON file.
    
    Args:
        json_file: Path to the JSON file
        is_nocontext: If True, expects 1 conversation per image_id with no goal key
        expected_goals: Number of goals expected per sample (default: 2, or 5 for resume5)
    
    Returns:
        dict: Statistics with counts for 0, 1, ..., expected_goals goals per sample (or 0, 1 for nocontext)
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if is_nocontext:
        # For nocontext: just count samples that have at least 1 conversation
        sample_counts = defaultdict(int)
        
        for item in data:
            image_id = item.get("image_id", "unknown")
            sample_counts[image_id] += 1
        
        # Count samples by number of conversations (0 or 1 expected)
        goal_counts = {0: 0, 1: 0, 2: 0}
        total_samples = 100  # Expected total
        
        for image_id, count in sample_counts.items():
            if count >= 1:
                goal_counts[1] += 1
            # Note: we use goal_counts[1] to represent "completed" samples for nocontext
        
        # Samples with 0 conversations = total - samples we've seen
        goal_counts[0] = total_samples - goal_counts[1]
        
        return {
            'total_samples': total_samples,
            'samples_with_goals': {img_id: set([f"conv_{i}"]) for img_id, i in sample_counts.items()},
            'goal_counts': goal_counts,
            'total_conversations': len(data),
            'is_nocontext': True
        }
    else:
        # Original logic: Group by image_id and collect unique goals
        sample_goals = defaultdict(set)
        
        for item in data:
            image_id = item.get("image_id", "unknown")
            context = item.get("context", {})
            goal = context.get("goal", "")
            
            if goal:  # Only count if goal exists
                sample_goals[image_id].add(goal)
        
        # Count samples by number of goals
        goal_counts = {i: 0 for i in range(expected_goals + 1)}
        total_samples = 100  # Expected total
        
        for image_id, goals in sample_goals.items():
            num_goals = len(goals)
            if num_goals <= expected_goals:
                goal_counts[num_goals] += 1
            else:
                # More than expected goals (unexpected, count as max)
                goal_counts[expected_goals] += 1
        
        # Samples with 0 goals = total - samples we've seen
        goal_counts[0] = total_samples - sum(goal_counts.values())
        
        return {
            'total_samples': total_samples,
            'samples_with_goals': sample_goals,
            'goal_counts': goal_counts,
            'total_conversations': len(data),
            'is_nocontext': False,
            'expected_goals': expected_goals
        }


def main():
    # Green models list - always copy these even if incomplete
    green_models = [
        "InternVL2-2B",
        "InternVL2-8B",
        "InternVL2_5-2B",
        "InternVL2_5-38B",
        "InternVL3-2B-Instruct",
        "InternVL3-8B-Instruct",
        "Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-72B-Instruct",
        "Qwen2.5-VL-7B-Instruct",
        "gemma-3-12b-it",
        "InternVL2_5-8B",
        "llava-1.5-13b-hf",
        "llava-1.5-7b-hf"
    ]
    
    parser = argparse.ArgumentParser(description='Check conversation generation progress')
    parser.add_argument('dir_path', type=str, 
                       help='Path to work_dirs output directory or JSON file')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed sample-by-sample breakdown')
    parser.add_argument('--csv', type=str, default=None,
                       help='Save statistics to CSV file (default: auto-generate based on dir_path)')
    parser.add_argument('--auto-csv', action='store_true',
                       help='Automatically save CSV with default name')
    parser.add_argument('--copy-completed', action='store_true',
                       help='Copy completed files (>95% complete) to {folder}_completed')
    parser.add_argument('--clean-duplicates', action='store_true',
                       help='Delete incomplete .json files if a _cache.json with same completeness exists')
    args = parser.parse_args()
    
    # Find all JSON files in the directory
    json_files = []
    if os.path.isdir(args.dir_path):
        for filename in os.listdir(args.dir_path):
            if filename.endswith('.json'): #and '_cache' not in filename
                json_files.append(os.path.join(args.dir_path, filename))
    elif os.path.isfile(args.dir_path) and args.dir_path.endswith('.json'):
        json_files = [args.dir_path]
    else:
        print(f"Error: {args.dir_path} is not a valid directory or JSON file")
        return
    
    if not json_files:
        print(f"No JSON files found in {args.dir_path}")
        return
    
    print(f"Found {len(json_files)} JSON file(s) in {args.dir_path}")
    print("=" * 80)
    
    # Check if this is a nocontext path
    is_nocontext = "nocontext" in args.dir_path.lower() or "_baseline" in args.dir_path.lower()
    if is_nocontext:
        print("📋 Detected 'nocontext' in path - using single conversation per sample mode")
        print("=" * 80)
    
    # Check if this is a resume5 path (expects 5 goals instead of 2)
    expected_goals = 5 if "resume5" in args.dir_path.lower() else 2
    if expected_goals == 5:
        print(f"📋 Detected 'resume5' in path - expecting {expected_goals} goals per sample")
        print("=" * 80)
    
    # Setup completed folder if copy is requested
    completed_folder = None
    completed_files = []
    if args.copy_completed and os.path.isdir(args.dir_path):
        dir_name = os.path.basename(args.dir_path.rstrip('/'))
        parent_dir = os.path.dirname(args.dir_path.rstrip('/'))
        completed_folder = os.path.join(parent_dir, f"{dir_name}_completed")
    
    # Store all results for CSV export and duplicate detection
    all_results = {}  # Changed to dict for easier lookup
    files_to_delete = []
    # Track files that are >95% but <100% complete
    near_complete_files = []  # List of (filename, completion_pct, filepath)
    # Track incomplete green model files for logging
    incomplete_green_files = []  # List of (filename, completion_pct, filepath, model_name)
    
    # Process each JSON file
    for json_file in sorted(json_files):
        filename = os.path.basename(json_file)
        print(f"\n📄 File: {filename}")
        print("-" * 80)
        
        try:
            stats = check_progress(json_file, is_nocontext=is_nocontext, expected_goals=expected_goals)
            
            print(f"Total conversations in file: {stats['total_conversations']}")
            print(f"Expected samples: {stats['total_samples']}")
            print()
            
            if is_nocontext:
                # For nocontext: only show completed (1 conv) vs not started (0 conv)
                print("Progress Summary:")
                print(f"  ✅ Completed (1 conversation): {stats['goal_counts'][1]:3d} samples")
                print(f"  ⏸️  Not started (0 conversations): {stats['goal_counts'][0]:3d} samples")
                print()
                
                # Calculate completion percentage
                completed = stats['goal_counts'][1]
                total_progress = completed / stats['total_samples'] * 100
                
                print(f"Overall Progress: {total_progress:.1f}% ({completed}/{stats['total_samples']} conversations)")
                
                # Store results for CSV
                result_data = {
                    'filename': filename,
                    '0_goals': stats['goal_counts'][0],
                    '1_goal': stats['goal_counts'][1],
                    '2_goals': 0,  # Not applicable for nocontext
                    'total_conversations': stats['total_conversations'],
                    'completion_percentage': round(total_progress, 2),
                    'filepath': json_file
                }
                all_results[filename] = result_data
                
                # Extract model name and check if it's in green list
                model_name = extract_model_name(filename)
                is_green_model = model_name in green_models
                is_cache_file = "_cache" in filename
                
                # Check if this file is completed (>95% threshold)
                is_compl, compl_pct = is_completed(stats['goal_counts'], is_nocontext=True, expected_goals=1, threshold=95)
                
                # Always copy cache files (will be renamed), or copy if green model, or copy if completed
                should_copy = False
                if args.copy_completed:
                    if is_cache_file:
                        # Always copy cache files
                        completed_files.append(json_file)
                        should_copy = True
                        # If cache file is from green model and incomplete, also log it
                        if is_green_model and not is_compl:
                            print(f"  ✨ Marked for copying (cache file from green model, incomplete: {compl_pct:.1f}%, will be renamed)")
                            incomplete_green_files.append((filename, compl_pct, json_file, model_name))
                        else:
                            print(f"  ✨ Marked for copying (cache file, will be renamed)")
                    elif is_green_model:
                        # Always copy green models
                        completed_files.append(json_file)
                        should_copy = True
                        if is_compl:
                            print(f"  ✨ Marked for copying (green model, {compl_pct:.1f}%)")
                        else:
                            print(f"  ✨ Marked for copying (green model, incomplete: {compl_pct:.1f}%)")
                            incomplete_green_files.append((filename, compl_pct, json_file, model_name))
                    elif is_compl:
                        completed_files.append(json_file)
                        should_copy = True
                        if compl_pct >= 100:
                            print(f"  ✨ Marked for copying to completed folder (100%)")
                        else:
                            print(f"  ✨ Marked for copying to completed folder ({compl_pct:.1f}%)")
                            near_complete_files.append((filename, compl_pct, json_file))
                
                if compl_pct >= 95 and compl_pct < 100 and not should_copy:
                    near_complete_files.append((filename, compl_pct, json_file))
            else:
                # Original logic for goal-based tracking
                print("Progress Summary:")
                # Show all goal counts up to expected_goals
                for i in range(expected_goals, -1, -1):
                    if i == expected_goals:
                        print(f"  ✅ {expected_goals} goals completed (finished): {stats['goal_counts'][i]:3d} samples")
                    elif i > 0:
                        print(f"  🔄 {i} goal(s) completed (in progress): {stats['goal_counts'][i]:3d} samples")
                    else:
                        print(f"  ⏸️  0 goals completed (not started): {stats['goal_counts'][i]:3d} samples")
                print()
                
                # Calculate completion percentage
                completed = stats['goal_counts'][expected_goals]
                in_progress = sum(stats['goal_counts'][i] * i for i in range(1, expected_goals))
                total_conversations = completed * expected_goals + in_progress
                total_expected = stats['total_samples'] * expected_goals
                total_progress = total_conversations / total_expected * 100
                
                print(f"Overall Progress: {total_progress:.1f}% ({total_conversations}/{total_expected} conversations)")
                
                # Store results for CSV
                result_data = {
                    'filename': filename,
                    '0_goals': stats['goal_counts'][0],
                    '1_goal': stats['goal_counts'][1],
                    '2_goals': stats['goal_counts'].get(2, 0),
                    'total_conversations': stats['total_conversations'],
                    'completion_percentage': round(total_progress, 2),
                    'filepath': json_file
                }
                # Only add goal fields up to expected_goals
                if expected_goals >= 3:
                    result_data['3_goals'] = stats['goal_counts'].get(3, 0)
                if expected_goals >= 4:
                    result_data['4_goals'] = stats['goal_counts'].get(4, 0)
                if expected_goals >= 5:
                    result_data['5_goals'] = stats['goal_counts'].get(5, 0)
                all_results[filename] = result_data
                
                # Extract model name and check if it's in green list
                model_name = extract_model_name(filename)
                is_green_model = model_name in green_models
                is_cache_file = "_cache" in filename
                
                # Check if this file is completed (>95% threshold)
                is_compl, compl_pct = is_completed(stats['goal_counts'], is_nocontext=False, expected_goals=expected_goals, threshold=95)
                
                # Always copy cache files (will be renamed), or copy if green model, or copy if completed
                should_copy = False
                if args.copy_completed:
                    if is_cache_file:
                        # Always copy cache files
                        completed_files.append(json_file)
                        should_copy = True
                        # If cache file is from green model and incomplete, also log it
                        if is_green_model and not is_compl:
                            print(f"  ✨ Marked for copying (cache file from green model, incomplete: {compl_pct:.1f}%, will be renamed)")
                            incomplete_green_files.append((filename, compl_pct, json_file, model_name))
                        else:
                            print(f"  ✨ Marked for copying (cache file, will be renamed)")
                    elif is_green_model:
                        # Always copy green models
                        completed_files.append(json_file)
                        should_copy = True
                        if is_compl:
                            print(f"  ✨ Marked for copying (green model, {compl_pct:.1f}%)")
                        else:
                            print(f"  ✨ Marked for copying (green model, incomplete: {compl_pct:.1f}%)")
                            incomplete_green_files.append((filename, compl_pct, json_file, model_name))
                    elif is_compl:
                        completed_files.append(json_file)
                        should_copy = True
                        if compl_pct >= 100:
                            print(f"  ✨ Marked for copying to completed folder (100%)")
                        else:
                            print(f"  ✨ Marked for copying to completed folder ({compl_pct:.1f}%)")
                            near_complete_files.append((filename, compl_pct, json_file))
                
                if compl_pct >= 95 and compl_pct < 100 and not should_copy:
                    near_complete_files.append((filename, compl_pct, json_file))
            
            if args.detailed:
                print("\nDetailed Breakdown:")
                samples_by_goal_count = defaultdict(list)
                for image_id, goals in stats['samples_with_goals'].items():
                    samples_by_goal_count[len(goals)].append(image_id)
                
                for num_goals in range(expected_goals, 0, -1):
                    if samples_by_goal_count[num_goals]:
                        print(f"\n  Samples with {num_goals} goal(s): {len(samples_by_goal_count[num_goals])}")
                        for image_id in sorted(samples_by_goal_count[num_goals])[:10]:
                            print(f"    - {image_id}")
                        if len(samples_by_goal_count[num_goals]) > 10:
                            print(f"    ... and {len(samples_by_goal_count[num_goals]) - 10} more")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    # Check for duplicate files with same completeness
    if args.clean_duplicates:
        print("\n" + "=" * 80)
        print("🔍 Checking for duplicate files with same completeness...")
        print("=" * 80)
        
        for filename, result in all_results.items():
            # Skip cache files
            if '_cache.json' in filename:
                continue
            
            # Skip if completed (>95%) - NEVER delete completed files
            goal_counts_dict = {}
            if is_nocontext:
                goal_counts_dict[1] = result.get('1_goal', 0)
            else:
                for i in range(expected_goals + 1):
                    key = f'{i}_goals' if i > 2 else ('1_goal' if i == 1 else ('2_goals' if i == 2 else '0_goals'))
                    goal_counts_dict[i] = result.get(key, 0)
            
            is_compl, _ = is_completed(goal_counts_dict, is_nocontext=is_nocontext, expected_goals=expected_goals, threshold=95)
            if is_compl:
                print(f"\n✓ Skipping {filename} (completed, will not delete)")
                continue
            
            # Look for corresponding cache file
            base_name = filename.replace('.json', '')
            cache_filename = f"{base_name}_cache.json"
            
            if cache_filename in all_results:
                cache_result = all_results[cache_filename]
                
                # Also check if cache is complete (>95%) - don't delete if either is complete
                cache_goal_counts = {}
                if is_nocontext:
                    cache_goal_counts[1] = cache_result.get('1_goal', 0)
                else:
                    for i in range(expected_goals + 1):
                        key = f'{i}_goals' if i > 2 else ('1_goal' if i == 1 else ('2_goals' if i == 2 else '0_goals'))
                        cache_goal_counts[i] = cache_result.get(key, 0)
                
                cache_is_compl, _ = is_completed(cache_goal_counts, is_nocontext=is_nocontext, expected_goals=expected_goals, threshold=95)
                if cache_is_compl:
                    print(f"\n✓ Skipping {filename} (cache version is completed)")
                    continue
                
                # Compare completeness - check all goal counts match
                goal_counts_match = True
                for i in range(expected_goals + 1):
                    key = f'{i}_goals' if i > 2 else ('1_goal' if i == 1 else ('2_goals' if i == 2 else '0_goals'))
                    if result.get(key, 0) != cache_result.get(key, 0):
                        goal_counts_match = False
                        break
                
                if goal_counts_match:
                    files_to_delete.append(result['filepath'])
                    print(f"\n🗑️  Found duplicate: {filename}")
                    if is_nocontext:
                        completed_key = '1_goal'
                    else:
                        completed_key = f'{expected_goals}_goals' if expected_goals > 2 else '2_goals'
                    print(f"   - {filename}: {result.get(completed_key, 0)} completed")
                    print(f"   - {cache_filename}: {cache_result.get(completed_key, 0)} completed")
                    print(f"   ➜ Marking {filename} for deletion (same completeness, both incomplete)")
                    
        # Delete marked files
        if files_to_delete:
            print(f"\n⚠️  Found {len(files_to_delete)} file(s) to delete")
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    filename = os.path.basename(file_path)
                    print(f"   ✓ Deleted: {filename}")
                    # Remove from all_results
                    if filename in all_results:
                        del all_results[filename]
                except Exception as e:
                    print(f"   ✗ Failed to delete {os.path.basename(file_path)}: {e}")
            print(f"\n✅ Successfully deleted {len(files_to_delete)} duplicate file(s)")
        else:
            print("\n✅ No duplicate files found")
    
    print("\n" + "=" * 80)
    
    # Auto-generate CSV path if requested or if --csv flag with no argument
    csv_path = args.csv
    if args.auto_csv and not csv_path:
        if os.path.isdir(args.dir_path):
            dir_name = os.path.basename(args.dir_path.rstrip('/'))
            csv_path = os.path.join(args.dir_path, f"{dir_name}_stats.csv")
        else:
            csv_path = args.dir_path.replace('.json', '_stats.csv')
    
    # Save to CSV if requested
    if csv_path and all_results:
        with open(csv_path, 'w', newline='') as csvfile:
            # Include all goal columns up to expected_goals
            fieldnames = ['filename', '0_goals', '1_goal']
            if expected_goals >= 2:
                fieldnames.append('2_goals')
            if expected_goals >= 3:
                fieldnames.append('3_goals')
            if expected_goals >= 4:
                fieldnames.append('4_goals')
            if expected_goals >= 5:
                fieldnames.append('5_goals')
            fieldnames.extend(['total_conversations', 'completion_percentage'])
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in all_results.values():
                # Remove filepath before writing to CSV
                csv_result = {k: v for k, v in result.items() if k != 'filepath'}
                writer.writerow(csv_result)
        
        print(f"\n✅ Statistics saved to: {csv_path}")
        print(f"   Exported {len(all_results)} file(s)")
    
    if args.copy_completed and completed_files and completed_folder:
        os.makedirs(completed_folder, exist_ok=True)
        
        files_to_copy = []
        rename_map = {}  # Maps src_file to new filename (if renaming needed)
        
        for src_file in completed_files:
            filename = os.path.basename(src_file)
            
            if "_cache" in filename:
                # Always copy cache files and rename them (remove _cache)
                non_cache_filename = filename.replace("_cache", "")
                files_to_copy.append(src_file)
                rename_map[src_file] = non_cache_filename
            else:
                # Non-cache file, copy as-is
                files_to_copy.append(src_file)

        print(f"\n📦 Copying {len(files_to_copy)} completed file(s) to: {completed_folder}")

        for src_file in files_to_copy:
            original_filename = os.path.basename(src_file)
            
            # Use renamed filename if in rename_map, otherwise use original
            filename = rename_map.get(src_file, original_filename)
            name_no_ext = os.path.splitext(filename)[0]

            # create per-file subfolder
            subfolder = os.path.join(completed_folder, name_no_ext)
            os.makedirs(subfolder, exist_ok=True)

            dst_file = os.path.join(subfolder, filename)
            shutil.copy2(src_file, dst_file)
            shutil.copy2(src_file, os.path.join(completed_folder, filename))

            if src_file in rename_map:
                print(f"   ✓ Copied & renamed: {original_filename} → {filename}")
            else:
                print(f"   ✓ Copied: {filename}")

        print(f"\n✅ Successfully copied {len(files_to_copy)} completed file(s)")
    
    # Write completeness.log for files that are >95% but <100% complete and incomplete green models
    log_entries = []
    if near_complete_files:
        log_entries.extend([(f, p, fp, None) for f, p, fp in near_complete_files])  # Add None for model_name
    if incomplete_green_files:
        log_entries.extend([(f, p, fp, m) for f, p, fp, m in incomplete_green_files])
    
    if log_entries and os.path.isdir(args.dir_path):
        log_path = os.path.join(args.dir_path, "completeness.log")
        with open(log_path, 'w') as f:
            f.write("Files with >95% but <100% completion and incomplete green models:\n")
            f.write("=" * 80 + "\n\n")
            # Sort by completion percentage (descending)
            sorted_entries = sorted(log_entries, key=lambda x: x[1], reverse=True)
            for entry in sorted_entries:
                filename, compl_pct, filepath, model_name = entry
                if model_name:
                    # Incomplete green model file
                    f.write(f"{filename} (green model {model_name}): {compl_pct:.2f}% complete\n")
                else:
                    # Regular near-complete file
                    f.write(f"{filename}: {compl_pct:.2f}% complete\n")
            f.write(f"\nTotal: {len(log_entries)} file(s)\n")
            if incomplete_green_files:
                f.write(f"Incomplete green models: {len(incomplete_green_files)} file(s)\n")
        print(f"\n📝 Wrote completeness.log: {len(log_entries)} file(s)")
        if incomplete_green_files:
            print(f"   Including {len(incomplete_green_files)} incomplete green model file(s)")


if __name__ == "__main__":
    main()