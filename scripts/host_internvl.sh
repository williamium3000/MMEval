#!/bin/bash
# Host InternVL via vLLM (reference: scripts/pope_infer_vllm_internvl.sh)
# Usage: bash scripts/host_internvl.sh [GPU] [PORT]
#   GPU default: 6, PORT default: 8007

set -e

# Configuration (override with env or args)
MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL3-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-InternVL3-8B-Instruct}"
GPUS=2,5,6,7
PORT="${2:-8007}"
GPU_UTIL="${GPU_UTIL:-0.25}"
LOG_FILE="tmp/vllm_${MODEL_NAME}_${PORT}.log"

mkdir -p tmp

eval "$(conda shell.bash hook)"
conda activate verl

export PYTHONPATH=./

echo "=================================================================================="
echo "Starting vLLM server for InternVL"
echo "=================================================================================="
echo "Model: $MODEL_PATH"
echo "GPUs:  $GPUS"
echo "Port:  $PORT"
echo "Log:   $LOG_FILE"
echo ""

echo "[$MODEL_NAME] Starting vLLM server on GPU(s) $GPUS..."
echo "[$MODEL_NAME] Note: InternVL may have video processing issues during initialization"
echo "[$MODEL_NAME] If startup fails, check $LOG_FILE for details"
(
    export CUDA_VISIBLE_DEVICES=$GPUS
    TORCH_SYMM_MEM_DISABLE_MULTICAST=1 python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --host 0.0.0.0 \
        --gpu-memory-utilization "$GPU_UTIL" \
        --trust-remote-code \
        --enforce-eager \
        --tensor-parallel-size $(echo "$GPUS" | tr ',' '\n' | wc -l) \
        > "$LOG_FILE" 2>&1
) &

VLLM_PID=$!
echo "[$MODEL_NAME] vLLM server started (PID: $VLLM_PID) on port $PORT"
echo ""

echo "Waiting for server to be ready..."
max_attempts=120
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "✓ Server is ready!"
        echo ""
        echo "=================================================================================="
        echo "Server is running!"
        echo "=================================================================================="
        echo "Endpoint: http://localhost:$PORT/v1/chat/completions"
        echo "Model ID: $MODEL_PATH"
        echo ""
        echo "To stop: kill $VLLM_PID"
        echo "Logs:    tail -f $LOG_FILE"
        echo "=================================================================================="
        break
    fi
    sleep 2
    attempt=$((attempt + 1))
    if [ $((attempt % 10)) -eq 0 ]; then
        echo "[$MODEL_NAME] Still waiting... (attempt $attempt/$max_attempts)"
    fi
done

if [ $attempt -eq $max_attempts ]; then
    echo "Error: Server failed to start within timeout"
    echo "Check logs: $LOG_FILE"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

wait $VLLM_PID
