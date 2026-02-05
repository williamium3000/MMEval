export PYTHONPATH=./
export PYTHONPATH=$PYTHONPATH:/raid/william/project/context-eval-mllm/infer/LLaVA/llava
export PYTHONPATH="/raid/william/project/context-eval-mllm/infer/LLaVA:${PYTHONPATH:-}"

#!/bin/bash
# Auto-process all *_completed folders recursively under work_dirs/vg
# Only process JSON files if model name exists as a subdirectory (ModelName/ModelName.json pattern)
eval "$(conda shell.bash hook)"
conda activate qwenvl

export PYTHONPATH=./

SAMPLE_NUM=100
START_IDX=0
export SAMPLE_NUM START_IDX

# Only process these models (exact match on basename without .json)
ALLOW_MODELS=(
    "Qwen2.5-VL-7B-Instruct"
    "llava-1.5-7b-hf"
    "InternVL3-8B-Instruct"
)

is_allowed_model() {
    local m="$1"
    for a in "${ALLOW_MODELS[@]}"; do
        if [ "$m" = "$a" ]; then
            return 0
        fi
    done
    return 1
}

process_file() {
    model_file="$1"
    model_dir="$(dirname "$model_file")"
    base="$(basename "$model_file" .json)"
    outfile="$model_dir/${base}_pope_converted.json"

    if ! is_allowed_model "$base"; then
        echo "[$(date +'%H:%M:%S')] Skipping (not in allowlist): $(basename "$model_file")"
        return 0
    fi

    # Check if model name exists as subdirectory
    model_subdir="$model_dir/$base"
    if [ ! -d "$model_subdir" ]; then
        echo "[$(date +'%H:%M:%S')] Skipping (no subdirectory): $(basename "$model_file") (expected $model_subdir)"
        return 0
    fi

    if [ -f "$outfile" ]; then
        echo "[$(date +'%H:%M:%S')] Skipping (already exists): $outfile"
        return 0
    fi

    echo "[$(date +'%H:%M:%S')] Starting: $(basename "$model_file") -> $outfile"

    python examiner/DSG_v2.py \
        --conv_script "$model_file" \
        --outfile "$outfile" \
        --extract_from_transcript \
        --sample_num "$SAMPLE_NUM" \
        --start_idx "$START_IDX" \
        --local

    echo "[$(date +'%H:%M:%S')] Done: $(basename "$model_file")"

    # Run extraction immediately after conversion
    if [ -f "$outfile" ]; then
        echo "[$(date +'%H:%M:%S')] Extracting dsg_qa: $outfile"
        python utils/extract_dsg_qa_questions.py "$outfile"
    fi
}

export -f process_file

# Base directory to search (default: work_dirs/vg)
BASE_DIR="${1:-work_dirs/vg}"

# Find all *_completed folders recursively
echo "[$(date +'%H:%M:%S')] Searching for *_completed folders under $BASE_DIR..."
completed_folders=$(find "$BASE_DIR" -type d -name "*_completed" | sort)

if [ -z "$completed_folders" ]; then
    echo "[$(date +'%H:%M:%S')] No *_completed folders found under $BASE_DIR"
    exit 0
fi

echo "[$(date +'%H:%M:%S')] Found $(echo "$completed_folders" | wc -l) *_completed folder(s)"

# Process each completed folder
for folder in $completed_folders; do
    echo ""
    echo "[$(date +'%H:%M:%S')] Processing folder: $folder"
    echo "=================================================================================="
    
    # Find JSON files in this folder (top level only, skip _cache and _pope_converted)
    for json_file in "$folder"/*.json; do
        # Skip if doesn't exist (glob didn't match)
        [ ! -f "$json_file" ] && continue
        # Skip cache and pope_converted files
        basename_file=$(basename "$json_file")
        [[ "$basename_file" == *_cache.json ]] && continue
        [[ "$basename_file" == *_pope_converted.json ]] && continue
        
        process_file "$json_file"
    done
done

echo ""
echo "[$(date +'%H:%M:%S')] All folders processed!"
