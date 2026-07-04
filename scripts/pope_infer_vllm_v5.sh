#!/bin/bash
# POPE inference pipeline for _v5 inputs only (3 models).
# Input:  *_pope_converted_v5.json
# Output: *_extracted_v5.json (extraction), *_pope_output_v5.json (inference)

set -e

SUFFIX="_v5"
BASE_DIR="${BASE_DIR:-work_dirs/vg/final_run_v18_gpt4o_completed}"

# Input files (one per model)
POPE_INPUTS=(
    #"$BASE_DIR/Qwen2.5-VL-7B-Instruct/Qwen2.5-VL-7B-Instruct_pope_converted_v5.json"
    #"$BASE_DIR/llava-1.5-7b-hf/llava-1.5-7b-hf_pope_converted_v5.json"
    "$BASE_DIR/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct_pope_converted_v5.json"
)

# Parse arguments
EXTRACT_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --extract)
            EXTRACT_ONLY=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

VLLM_PORT_START=8005
MAX_CONCURRENT="${MAX_CONCURRENT:-24}"

MODELS=(
    # "Qwen2.5-VL-7B-Instruct"
    # "llava-1.5-7b-hf"
    "Qwen3-VL-8B-Instruct"
)

declare -A MODEL_PATHS=(
    ["Qwen2.5-VL-7B-Instruct"]="Qwen/Qwen2.5-VL-7B-Instruct"
    ["llava-1.5-7b-hf"]="llava-hf/llava-1.5-7b-hf"
    ["Qwen3-VL-8B-Instruct"]="Qwen/Qwen3-VL-8B-Instruct"
)

CONDA_ENV="qwenvl3_vllm"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export PYTHONPATH=./

echo "=================================================================================="
echo "POPE Inference Pipeline (v5): 3 models, all outputs with _v5 suffix"
echo "=================================================================================="
echo "Inputs: ${POPE_INPUTS[@]}"
echo ""

# Step 1: Extract questions from the 2 _pope_converted_v5.json files
echo "Step 1: Extracting questions from _pope_converted_v5.json files..."
echo "=================================================================================="

for pope_file in "${POPE_INPUTS[@]}"; do
    if [ ! -f "$pope_file" ]; then
        echo "Skip (missing): $pope_file"
        continue
    fi
    model_dir=$(dirname "$pope_file")
    base_name=$(basename "$pope_file" "_pope_converted_v5.json")
    extracted_file="${model_dir}/${base_name}_extracted_v5.json"

    if [ -f "$extracted_file" ]; then
        echo "Already extracted: $extracted_file"
    else
        echo "Extracting: $pope_file -> $extracted_file"
        python utils/extract_dsg_qa_questions.py "$pope_file" --output "$extracted_file"
    fi
done

if [ "$EXTRACT_ONLY" = true ]; then
    echo ""
    echo "Extract-only mode: exiting after extraction."
    exit 0
fi

echo ""
echo "Step 2: Starting vLLM servers and running inference..."
echo "=================================================================================="

process_model() {
    local model_idx=$1
    local model="${MODELS[$model_idx]}"
    local gpu=$((1 + model_idx))
    local port=$((VLLM_PORT_START + model_idx))
    local model_path="${MODEL_PATHS[$model]}"

    if [ -z "$model_path" ]; then
        echo "[$model] Warning: Model identifier not set"
        return 1
    fi

    echo ""
    echo "[$model] Starting on GPU $gpu, port $port"
    echo "[$model] Model path: $model_path"

    # Only _extracted_v5.json for this model
    local extracted_files=$(find "$BASE_DIR" -type f -name "*${model}*_extracted_v5.json" 2>/dev/null || true)

    if [ -z "$extracted_files" ]; then
        echo "[$model] No *_extracted_v5.json found, skipping..."
        return 1
    fi

    echo "[$model] Starting vLLM server on GPU $gpu..."

    local vllm_args=(
        "--model" "$model_path"
        "--port" "$port"
        "--host" "0.0.0.0"
        "--gpu-memory-utilization" "0.6"
    )

    case "$model" in
        "InternVL3-8B-Instruct"|"Qwen3-VL-8B-Instruct")
            vllm_args+=("--trust-remote-code")
            ;;
    esac

    mkdir -p tmp
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        unset VLLM_API_KEY
        python -m vllm.entrypoints.openai.api_server \
            "${vllm_args[@]}" \
            > "tmp/vllm_${model}_${port}_v5.log" 2>&1
    ) &

    local VLLM_PID=$!
    echo "[$model] vLLM server started (PID: $VLLM_PID) on port $port"

    echo "[$model] Waiting for server to be ready (hosting model)..."
    local max_attempts=300
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" 2>/dev/null | grep -q 200; then
            if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
                echo "[$model] Server is ready (model hosted)."
                break
            fi
        fi
        sleep 2
        attempt=$((attempt + 1))
        if [ $((attempt % 15)) -eq 0 ] && [ $attempt -gt 0 ]; then
            echo "[$model] Still waiting for hosting... ($((attempt * 2))s)"
        fi
    done

    if [ $attempt -eq $max_attempts ]; then
        echo "[$model] Error: Server failed to start within timeout"
        kill $VLLM_PID 2>/dev/null || true
        return 1
    fi

    local file_array=()
    while IFS= read -r extracted_file; do
        if [ -f "$extracted_file" ]; then
            file_array+=("$extracted_file")
        fi
    done <<< "$extracted_files"

    if [ ${#file_array[@]} -eq 0 ]; then
        echo "[$model] No files to process"
    else
        echo "[$model] Processing ${#file_array[@]} file(s)..."

        local active_jobs=0
        for extracted_file in "${file_array[@]}"; do
            local model_dir=$(dirname "$extracted_file")
            local base_name=$(basename "$extracted_file" "_extracted_v5.json")
            local output_file="${model_dir}/${base_name}_pope_output_v5.json"

            if [ -f "$output_file" ]; then
                echo "[$model] Skipping (output exists): $(basename "$output_file")"
                continue
            fi

            while [ $active_jobs -ge $MAX_CONCURRENT ]; do
                wait -n
                active_jobs=$((active_jobs - 1))
            done

            echo "[$model] Processing: $(basename "$extracted_file") -> $(basename "$output_file")"

            python infer/infer_vllm_for_pope.py \
                "$extracted_file" \
                --api-url "http://localhost:$port/v1/chat/completions" \
                --model-name "$model_path" \
                --output "$output_file" &

            active_jobs=$((active_jobs + 1))
        done

        wait
        echo "[$model] All files processed"
    fi

    echo "[$model] Stopping vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    sleep 5

    if kill -0 $VLLM_PID 2>/dev/null; then
        kill -9 $VLLM_PID 2>/dev/null || true
    fi

    echo "[$model] Completed"
}

echo "Starting 3 models in parallel on GPUs 1, 2, 3..."
pids=()
for i in "${!MODELS[@]}"; do
    process_model $i &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait $pid
done

echo ""
echo "=================================================================================="
echo "All models processed (v5). Outputs: *_pope_output_v5.json"
echo "=================================================================================="
