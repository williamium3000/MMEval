#!/bin/bash
# Script to run POPE inference pipeline for InternVL3-8B-Instruct only using vLLM
# 1. Extract questions from _pope_converted.json files for InternVL3
# 2. Start vLLM server for InternVL3
# 3. Run inference for InternVL3
# 4. Output _pope_output.json files

set -e

# Configuration
BASE_DIR="${1:-work_dirs/vg}"
VLLM_PORT=8002
GPU=6
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

# Model configuration
MODEL="InternVL3-8B-Instruct"
MODEL_PATH="OpenGVLab/InternVL3-8B-Instruct"

# Conda environment
CONDA_ENV="verl"  # Adjust to your environment

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export PYTHONPATH=./

echo "=================================================================================="
echo "POPE Inference Pipeline with vLLM - InternVL3-8B-Instruct Only"
echo "=================================================================================="
echo "Base directory: $BASE_DIR"
echo "Model: $MODEL"
echo "GPU: $GPU"
echo "Port: $VLLM_PORT"
echo ""

# Step 1: Extract questions from all _pope_converted.json files for InternVL3
echo "Step 1: Extracting questions from ALL _pope_converted.json files for $MODEL..."
echo "=================================================================================="

# Collect all files that need extraction first (avoid subshell issues)
declare -a files_to_extract=()

while IFS= read -r pope_file; do
    model_dir=$(dirname "$pope_file")
    base_name=$(basename "$pope_file" "_pope_converted.json")
    
    # Check if this file is for InternVL3
    if [[ "$base_name" != *"$MODEL"* ]] && [[ "$pope_file" != *"$MODEL"* ]]; then
        echo "Skipping (not InternVL3): $pope_file"
        continue
    fi
    
    extracted_file="${model_dir}/${base_name}_extracted.json"
    
    if [ -f "$extracted_file" ]; then
        echo "Already extracted: $extracted_file"
    else
        files_to_extract+=("$pope_file|$extracted_file")
    fi
done < <(find "$BASE_DIR" -type f -name "*_pope_converted.json")

# Now extract all files sequentially
if [ ${#files_to_extract[@]} -gt 0 ]; then
    echo ""
    echo "Extracting ${#files_to_extract[@]} files..."
    for file_pair in "${files_to_extract[@]}"; do
        IFS='|' read -r pope_file extracted_file <<< "$file_pair"
        echo "Extracting: $pope_file -> $extracted_file"
        python utils/extract_dsg_qa_questions.py "$pope_file" --output "$extracted_file"
    done
    echo "All extractions completed!"
else
    echo "All files already extracted."
fi

echo ""
echo "Step 2: Starting vLLM server and running inference..."
echo "=================================================================================="

# Find all extracted files for InternVL3
extracted_files=$(find "$BASE_DIR" -type f -name "*${MODEL}*_extracted.json" 2>/dev/null || true)

if [ -z "$extracted_files" ]; then
    echo "No extracted files found for $MODEL, exiting..."
    exit 1
fi

# Start vLLM server in background on GPU 6
echo "[$MODEL] Starting vLLM server on GPU $GPU..."
echo "[$MODEL] Note: InternVL3 may have video processing issues during initialization"
echo "[$MODEL] If startup fails, check tmp/vllm_${MODEL}_${VLLM_PORT}.log for details"
(
    export CUDA_VISIBLE_DEVICES=$GPU
    # Try with enforce_eager to avoid compilation issues
    # Also disable video processing if possible by using image-only mode
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port $VLLM_PORT \
        --host 0.0.0.0 \
        --gpu-memory-utilization 0.6 \
        --trust-remote-code \
        --enforce-eager \
        > "tmp/vllm_${MODEL}_${VLLM_PORT}.log" 2>&1
) &

VLLM_PID=$!
echo "[$MODEL] vLLM server started (PID: $VLLM_PID) on port $VLLM_PORT"

# Wait for server to be ready
echo "[$MODEL] Waiting for server to be ready..."
max_attempts=120  # Increased timeout for InternVL3
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "[$MODEL] Server is ready!"
        break
    fi
    sleep 2
    attempt=$((attempt + 1))
    if [ $((attempt % 10)) -eq 0 ]; then
        echo "[$MODEL] Still waiting... (attempt $attempt/$max_attempts)"
    fi
done

if [ $attempt -eq $max_attempts ]; then
    echo "[$MODEL] Error: Server failed to start within timeout"
    echo "[$MODEL] Check log: tmp/vllm_${MODEL}_${VLLM_PORT}.log"
    echo ""
    echo "[$MODEL] Common InternVL3 issues:"
    echo "  1. Video processing dtype error (TypeError: Cannot handle this data type: (1, 1, 3), <i8)"
    echo "     This is a known vLLM/InternVL3 compatibility issue"
    echo "  2. Try updating vLLM: pip install --upgrade vllm"
    echo "  3. Or use a different vLLM version compatible with InternVL3"
    echo ""
    echo "[$MODEL] Last 20 lines of log:"
    tail -n 20 "tmp/vllm_${MODEL}_${VLLM_PORT}.log" || true
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Process each extracted file for InternVL3 concurrently
# Collect files into array first
file_array=()
while IFS= read -r extracted_file; do
    if [ -f "$extracted_file" ]; then
        file_array+=("$extracted_file")
    fi
done <<< "$extracted_files"

if [ ${#file_array[@]} -eq 0 ]; then
    echo "[$MODEL] No files to process"
else
    echo "[$MODEL] Processing ${#file_array[@]} files with $MAX_CONCURRENT concurrent workers..."
    
    # Process files with proper concurrency control
    active_jobs=0
    for extracted_file in "${file_array[@]}"; do
        model_dir=$(dirname "$extracted_file")
        base_name=$(basename "$extracted_file" "_extracted.json")
        output_file="${model_dir}/${base_name}_pope_output.json"
        
        if [ -f "$output_file" ]; then
            echo "[$MODEL] Skipping (output exists): $(basename "$output_file")"
            continue
        fi
        
        # Wait if we've reached max concurrent jobs
        while [ $active_jobs -ge $MAX_CONCURRENT ]; do
            wait -n  # Wait for any job to complete
            active_jobs=$((active_jobs - 1))
        done
        
        echo "[$MODEL] Processing: $(basename "$extracted_file") -> $(basename "$output_file")"
        
        # Run inference in background
        python infer/infer_vllm_for_pope.py \
            "$extracted_file" \
            --api-url "http://localhost:$VLLM_PORT/v1/chat/completions" \
            --model-name "$MODEL_PATH" \
            --output "$output_file" &
        
        active_jobs=$((active_jobs + 1))
    done
    
    # Final wait to ensure all jobs are complete
    wait
    echo "[$MODEL] All files processed"
fi

# Stop vLLM server
echo "[$MODEL] Stopping vLLM server (PID: $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null || true
sleep 5

# Ensure it's stopped
if kill -0 $VLLM_PID 2>/dev/null; then
    kill -9 $VLLM_PID 2>/dev/null || true
fi

echo ""
echo "=================================================================================="
echo "[$MODEL] Completed!"
echo "=================================================================================="
