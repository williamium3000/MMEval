#!/bin/bash
# Script to run POPE inference pipeline for 3 VLM models using vLLM
# 1. Extract questions from _pope_converted.json files
# 2. Start vLLM server for each model
# 3. Run inference for each model
# 4. Output _pope_output.json files

set -e

# Parse arguments
EXTRACT_ONLY=false
BASE_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --extract)
            EXTRACT_ONLY=true
            shift
            ;;
        *)
            if [ -z "$BASE_DIR" ]; then
                BASE_DIR="$1"
            fi
            shift
            ;;
    esac
done

# Configuration
BASE_DIR="${BASE_DIR:-work_dirs/vg/final_run_v18_gpt4o_completed}"
VLLM_PORT_START=8000
# Maximum concurrent file processing jobs
# For H200 (141GB, 60% utilization): recommended 24-32
# Adjust based on GPU utilization and memory usage
MAX_CONCURRENT="${MAX_CONCURRENT:-24}"

# Models to process (must match pope_convert.sh ALLOW_MODELS)
MODELS=(
    "Qwen2.5-VL-7B-Instruct"
    "llava-1.5-7b-hf"
)

# vLLM model identifiers (HuggingFace Hub)
# vLLM will automatically download from HuggingFace if not cached locally
declare -A MODEL_PATHS=(
    ["Qwen2.5-VL-7B-Instruct"]="Qwen/Qwen2.5-VL-7B-Instruct"
    ["llava-1.5-7b-hf"]="llava-hf/llava-1.5-7b-hf"
)

# Models run in parallel: one model per GPU (GPUs 4, 5, 6)
# Each vLLM instance uses a single GPU and handles 8 concurrent requests

# Conda environment
CONDA_ENV="verl"  # Adjust to your environment

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export PYTHONPATH=./

echo "=================================================================================="
echo "POPE Inference Pipeline with vLLM"
echo "=================================================================================="
echo "Base directory: $BASE_DIR"
echo "Models: ${MODELS[@]}"
echo ""

# Step 1: Extract questions from all _pope_converted.json files
echo "Step 1: Extracting questions from ALL _pope_converted.json files..."
echo "=================================================================================="

# Collect all files that need extraction first (avoid subshell issues)
declare -a files_to_extract=()

while IFS= read -r pope_file; do
    model_dir=$(dirname "$pope_file")
    base_name=$(basename "$pope_file" "_pope_converted.json")
    
    # Check if model is in our list
    is_target_model=false
    for model in "${MODELS[@]}"; do
        if [[ "$base_name" == *"$model"* ]] || [[ "$pope_file" == *"$model"* ]]; then
            is_target_model=true
            break
        fi
    done
    
    if [ "$is_target_model" = false ]; then
        echo "Skipping (not in target models): $pope_file"
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

# If --extract: only run extraction (where _pope_converted exists), then exit
if [ "$EXTRACT_ONLY" = true ]; then
    echo ""
    echo "Extract-only mode: exiting after extraction."
    exit 0
fi

echo ""
echo "Step 2: Starting vLLM servers and running inference..."
echo "=================================================================================="

# Function to process a single model on a specific GPU
process_model() {
    local model_idx=$1
    local model="${MODELS[$model_idx]}"
    # Hardcoded GPU assignment: 4, 5, 6
    local gpu=$((4 + model_idx))
    local port=$((VLLM_PORT_START + model_idx))
    local model_path="${MODEL_PATHS[$model]}"
    
    if [ -z "$model_path" ]; then
        echo "[$model] Warning: Model identifier not set"
        return 1
    fi
    
    echo ""
    echo "[$model] Starting on GPU $gpu, port $port"
    echo "[$model] Model path: $model_path"
    
    # Find all extracted files for this model
    local extracted_files=$(find "$BASE_DIR" -type f -name "*${model}*_extracted.json" 2>/dev/null || true)
    
    if [ -z "$extracted_files" ]; then
        echo "[$model] No extracted files found, skipping..."
        return 1
    fi
    
    # Start vLLM server in background on single GPU
    # Export CUDA_VISIBLE_DEVICES in a subshell to ensure worker processes inherit it
    echo "[$model] Starting vLLM server on GPU $gpu..."
    
    # Build vLLM command with optional trust_remote_code flag
    local vllm_args=(
        "--model" "$model_path"
        "--port" "$port"
        "--host" "0.0.0.0"
        "--gpu-memory-utilization" "0.6"
    )
    
    # Add trust_remote_code if needed
    # Use case statement to avoid bash arithmetic expansion issues with associative arrays
    case "$model" in
        "InternVL3-8B-Instruct")
            vllm_args+=("--trust-remote-code")
            ;;
    esac
    
    # Run vLLM in a subshell with exported CUDA_VISIBLE_DEVICES
    # This ensures worker processes spawned by vLLM can see the GPU
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        python -m vllm.entrypoints.openai.api_server \
            "${vllm_args[@]}" \
            > "tmp/vllm_${model}_${port}.log" 2>&1
    ) &
    
    local VLLM_PID=$!
    echo "[$model] vLLM server started (PID: $VLLM_PID) on port $port"
    
    # Wait for server to be ready
    echo "[$model] Waiting for server to be ready..."
    local max_attempts=60
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "[$model] Server is ready!"
            break
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -eq $max_attempts ]; then
        echo "[$model] Error: Server failed to start within timeout"
        kill $VLLM_PID 2>/dev/null || true
        return 1
    fi
    
    # Process each extracted file for this model concurrently
    # Collect files into array first
    local file_array=()
    while IFS= read -r extracted_file; do
        if [ -f "$extracted_file" ]; then
            file_array+=("$extracted_file")
        fi
    done <<< "$extracted_files"
    
    if [ ${#file_array[@]} -eq 0 ]; then
        echo "[$model] No files to process"
    else
        echo "[$model] Processing ${#file_array[@]} files with $MAX_CONCURRENT concurrent workers..."
        
        # Process files with proper concurrency control
        local active_jobs=0
        for extracted_file in "${file_array[@]}"; do
            local model_dir=$(dirname "$extracted_file")
            local base_name=$(basename "$extracted_file" "_extracted.json")
            local output_file="${model_dir}/${base_name}_pope_output.json"
            
            if [ -f "$output_file" ]; then
                echo "[$model] Skipping (output exists): $(basename "$output_file")"
                continue
            fi
            
            # Wait if we've reached max concurrent jobs
            while [ $active_jobs -ge $MAX_CONCURRENT ]; do
                wait -n  # Wait for any job to complete
                active_jobs=$((active_jobs - 1))
            done
            
            echo "[$model] Processing: $(basename "$extracted_file") -> $(basename "$output_file")"
            
            # Run inference in background
            python infer/infer_vllm_for_pope.py \
                "$extracted_file" \
                --api-url "http://localhost:$port/v1/chat/completions" \
                --model-name "$model_path" \
                --output "$output_file" &
            
            active_jobs=$((active_jobs + 1))
        done
        
        # Final wait to ensure all jobs are complete
        wait
        echo "[$model] All files processed"
    fi
    
    # Stop vLLM server
    echo "[$model] Stopping vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    sleep 5
    
    # Ensure it's stopped
    if kill -0 $VLLM_PID 2>/dev/null; then
        kill -9 $VLLM_PID 2>/dev/null || true
    fi
    
    echo "[$model] Completed"
}

# Process all 3 models in parallel (one per GPU)
echo "Starting 3 models in parallel on GPUs 4, 5, 6..."
pids=()
for i in "${!MODELS[@]}"; do
    process_model $i &
    pids+=($!)
done

# Wait for all models to complete
for pid in "${pids[@]}"; do
    wait $pid
done

echo ""
echo "=================================================================================="
echo "All models processed!"
echo "=================================================================================="
