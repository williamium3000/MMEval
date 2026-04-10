#!/bin/bash
# Script to run verify_only on all *_pope_converted.json files
# Processes files at max depth 1 in work_dirs/vg/final_run_v18_gpt4o_completed
# Saves results in indir/{model_name}/

export PYTHONPATH=./
export PYTHONPATH=$PYTHONPATH:/raid/william/project/context-eval-mllm/infer/LLaVA/llava
export PYTHONPATH="/raid/william/project/context-eval-mllm/infer/LLaVA:${PYTHONPATH:-}"

eval "$(conda shell.bash hook)"
conda activate qwenvl

export PYTHONPATH=./

# Parse arguments
USE_LOCAL=false
INPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            USE_LOCAL=true
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

# Set input directory (default)
INPUT_DIR="${INPUT_DIR:-work_dirs/vg/final_run_v18_gpt4o_completed}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

echo "=================================================================================="
echo "Verify-only mode: Processing *_pope_converted.json files"
echo "=================================================================================="
echo "Input directory: $INPUT_DIR"
[ "$USE_LOCAL" = true ] && echo "Using LOCAL vLLM server (port 8004)" || echo "Using REMOTE API"
echo ""

# Find all *_pope_converted.json files at max depth 1
mapfile -t pope_files < <(find "$INPUT_DIR" -maxdepth 1 -name "*_pope_converted.json" -type f | sort)

if [ ${#pope_files[@]} -eq 0 ]; then
    echo "No *_pope_converted.json files found in $INPUT_DIR"
    exit 0
fi

echo "Found ${#pope_files[@]} file(s) to process"
echo ""

# Process each file
processed=0
skipped=0

for pope_file in "${pope_files[@]}"; do
    filename=$(basename "$pope_file")
    # Extract model name: remove _pope_converted.json suffix
    model_name="${filename%_pope_converted.json}"
    
    # Create output directory: indir/{model_name}/
    output_dir="$INPUT_DIR/$model_name"
    mkdir -p "$output_dir"
    
    # Output file: save as {model_name}_pope_verified.json in the subdirectory
    output_file="$output_dir/${model_name}_pope_verified.json"
    
    # Skip if already processed (output exists)
    if [ -f "$output_file" ]; then
        echo "[$(date +'%H:%M:%S')] ⏭️  Skipping (already exists): $filename -> $output_file"
        ((skipped++))
        continue
    fi
    
    # Check: does file have any dsg_qa? (check up to 50 samples for efficiency)
    has_dsg_qa=$(python3 -c "
import json
try:
    with open('$pope_file', 'r') as f:
        data = json.load(f)
    # Check up to 50 samples (or all if fewer)
    check_samples = min(50, len(data))
    for sample in data[:check_samples]:
        for turn in sample.get('conversations', []):
            if turn.get('dsg_qa') and len(turn.get('dsg_qa', [])) > 0:
                print('yes')
                exit(0)
    print('no')
except Exception as e:
    print('no')
" 2>/dev/null)
    
    if [ "$has_dsg_qa" != "yes" ]; then
        echo "[$(date +'%H:%M:%S')] ⚠️  Skipping (no dsg_qa found): $filename"
        echo "  This file has no dsg_qa fields to verify. Re-run conversion to generate DSG questions."
        ((skipped++))
        echo ""
        continue
    fi
    
    echo "[$(date +'%H:%M:%S')] 🔍 Processing: $filename"
    echo "  Input:  $pope_file"
    echo "  Output: $output_file"
    
    # Build command
    cmd="python examiner/DSG_v2.py \
        --conv_script \"$pope_file\" \
        --outfile \"$output_file\" \
        --verify_only \
        --batch_size 5"
    
    if [ "$USE_LOCAL" = true ]; then
        cmd="$cmd --local"
    fi
    
    # Run verification
    if eval "$cmd"; then
        echo "[$(date +'%H:%M:%S')] ✅ Completed: $filename"
        ((processed++))
    else
        echo "[$(date +'%H:%M:%S')] ❌ Failed: $filename"
    fi
    echo ""
done

echo "=================================================================================="
echo "Summary:"
echo "  Processed: $processed"
echo "  Skipped:   $skipped"
echo "  Total:     ${#pope_files[@]}"
echo "=================================================================================="
