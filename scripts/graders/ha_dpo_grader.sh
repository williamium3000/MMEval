#!/bin/bash

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=4
unset all_proxy

source /raid/miniconda3/etc/profile.d/conda.sh
conda activate coneval-easydetect

# Maximum number of parallel jobs
MAX_JOBS=10

# Parse arguments
USE_LOCAL=false
USE_LOCAL2=false
INPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            USE_LOCAL=true
            shift
            ;;
        --local2)
            USE_LOCAL2=true
            shift
            ;;
        *)
            if [ -z "$INPUT_DIR" ]; then
                INPUT_DIR="$1"
            fi
            shift
            ;;
    esac
done

# Set input directory (default if not provided)
INPUT_DIR="${INPUT_DIR:-work_dirs/vg/final_run_v18_gpt4o_completed}"

# Set API URL and key based on --local or --local2 flag
if [ "$USE_LOCAL2" = true ]; then
    API_URL="http://localhost:8004/v1"
    API_KEY=""  # Local vLLM doesn't require API key
    GPT_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
    echo "Using LOCAL2 vLLM server at $API_URL (port 8004)"
elif [ "$USE_LOCAL" = true ]; then
    API_URL="http://localhost:8003/v1"
    API_KEY=""  # Local vLLM doesn't require API key
    GPT_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
    echo "Using LOCAL vLLM server at $API_URL (port 8003)"
else
    API_URL=""  # Use default OpenAI API
    API_KEY=""  # Will use env var OPENAI_API_KEY
    GPT_MODEL="gpt-4o"
    echo "Using REMOTE API (OpenAI)"
fi

# Get all model files into array
mapfile -t model_files < <(find "$INPUT_DIR" -maxdepth 1 -name "*.json" -type f)
total_files=${#model_files[@]}

echo "Input directory: $INPUT_DIR"
echo "Found $total_files model files to evaluate"
echo "Running with $MAX_JOBS parallel jobs"
echo "========================================"
echo ""

# Counter for completed jobs
completed=0
skipped=0
# Array to track running PIDs
declare -a running_pids=()

for model_file in "${model_files[@]}"; do
    dir=$(dirname "$model_file")
    base=$(basename "$model_file" .json)
    outdir="${dir}/${base}"
    log_file="${outdir}/ha_dpo_grader.log"
    
    # Create output directory
    mkdir -p "$outdir"

    # Wait if we have MAX_JOBS running - check and clean up finished PIDs
    while [ ${#running_pids[@]} -ge $MAX_JOBS ]; do
        # Check which PIDs are still running
        new_pids=()
        for pid in "${running_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            else
                wait "$pid" 2>/dev/null
                ((completed++))
            fi
        done
        running_pids=("${new_pids[@]}")
        sleep 0.5
    done

    echo "🚀 [$((completed + ${#running_pids[@]} + skipped))/$total_files] Starting: $base"

    # Run in background and track PID
    (
        if [ -n "$API_URL" ]; then
            python grader/ha_dpo_grader/eval.py \
                "$model_file" \
                --outdir "$outdir" \
                --api-url "$API_URL" \
                --api-key "$API_KEY" \
                --gpt-model "$GPT_MODEL"
        else
            python grader/ha_dpo_grader/eval.py \
                "$model_file" \
                --outdir "$outdir" \
                --gpt-model "$GPT_MODEL"
        fi
        
        echo "✅ Completed: $base"
    ) > "$log_file" 2>&1 &
    
    # Track the PID
    running_pids+=($!)
done

# Wait for all remaining background jobs to complete
for pid in "${running_pids[@]}"; do
    wait "$pid" 2>/dev/null && ((completed++))
done

echo ""
echo "========================================"
echo "Summary:"
echo "  Evaluated: $completed"
echo "  Skipped:   $skipped"
echo "  Total:     $total_files"
echo "========================================"