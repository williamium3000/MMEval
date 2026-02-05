#!/usr/bin/env python3
"""
Auto-check source folders derived from *_completed (name with _completed removed).
For each X_completed under work_dirs/vg we scan only X, not X_completed.
Report a CSV: rows = model names (union across folders), columns = folder names.
Cell value = completed samples count (0–100) for that model in that folder.
Read-only; no copy/clean or other operations.
"""

import os
import csv
import sys

# Allow import from utils.check_output when run as script
_this_dir = os.path.dirname(os.path.abspath(__file__))
_proj = os.path.dirname(_this_dir)
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from utils.check_output import check_progress


VG_ROOT = "/raid/william/project/context-eval-mllm/work_dirs/vg"

# Necessary models to flag and prioritize
GREEN_MODELS = [
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


def get_folders_to_scan(root=VG_ROOT):
    """
    Return list of (folder_name, folder_path) to scan.
    For each *_completed folder we include only the source folder:
      X (path with _completed removed). Do not include the *_completed folders.
    """
    seen = set()
    out = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path) or not name.endswith("_completed"):
            continue
        source_name = name.replace("_completed", "")
        source_path = os.path.join(root, source_name)
        if source_name not in seen and os.path.isdir(source_path):
            seen.add(source_name)
            out.append((source_name, source_path))
    return out


def get_model_to_json_path(folder_path):
    """
    Map model_name -> path to the JSON file to use for that model.
    Prefer Model/Model.json; else Model.json. Skip _cache and _pope_converted.
    """
    model_to_path = {}
    try:
        entries = os.listdir(folder_path)
    except OSError:
        return model_to_path

    for name in entries:
        path = os.path.join(folder_path, name)
        if name.endswith("_cache.json") or name.endswith("_pope_converted.json"):
            continue
        if name.endswith(".csv"):
            continue
        if os.path.isfile(path) and name.endswith(".json"):
            model = name[:-5]  # strip .json
            model_to_path[model] = path
        if os.path.isdir(path):
            inner = os.path.join(path, name + ".json")
            if os.path.isfile(inner):
                model_to_path[name] = inner
    return model_to_path


def get_folder_options(folder_name, folder_path):
    """Derive is_nocontext and expected_goals from folder name/path."""
    path_lower = (folder_name + " " + folder_path).lower()
    is_nocontext = "nocontext" in path_lower or "baseline" in path_lower
    expected_goals = 5 if "resume5" in path_lower else 2
    return is_nocontext, expected_goals


def run():
    folders = get_folders_to_scan()
    if not folders:
        print(f"No source folders (from *_completed) under {VG_ROOT}")
        return

    # folder_name -> model_name -> number (completed samples)
    data = {}  # folder -> { model -> value }
    all_models = set()

    for folder_name, folder_path in folders:
        if not os.path.isdir(folder_path):
            continue
        is_nocontext, expected_goals = get_folder_options(folder_name, folder_path)
        model_to_path = get_model_to_json_path(folder_path)
        data[folder_name] = {}
        for model_name, json_path in model_to_path.items():
            all_models.add(model_name)
            try:
                stats = check_progress(json_path, is_nocontext=is_nocontext, expected_goals=expected_goals)
                if is_nocontext:
                    num = stats["goal_counts"].get(1, 0)
                else:
                    num = stats["goal_counts"].get(expected_goals, 0)
                data[folder_name][model_name] = num
            except Exception as e:
                data[folder_name][model_name] = f"err:{e}"

    folder_names = [f[0] for f in folders]
    
    # Sort models: green_models first (in order), then others alphabetically
    green_models_set = set(GREEN_MODELS)
    green_models_sorted = [m for m in GREEN_MODELS if m in all_models]
    other_models_sorted = sorted(all_models - green_models_set)
    model_names = green_models_sorted + other_models_sorted

    # Write CSV: header = model, folder1, folder2, ... ; rows = model, v1, v2, ...
    out_path = os.path.join(VG_ROOT, "overall_completed_report.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + folder_names)
        for model in model_names:
            row = [model]
            for fn in folder_names:
                v = data.get(fn, {}).get(model, "")
                row.append(v)
            w.writerow(row)

    print(f"Wrote {out_path}")
    print(f"Rows (models): {len(model_names)}, Columns (folders): {len(folder_names)}")


if __name__ == "__main__":
    run()
