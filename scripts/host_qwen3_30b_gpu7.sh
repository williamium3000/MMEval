#!/bin/bash
# Script to host Qwen3-30B-A3B-Instruct-2507 on GPU 7 using vLLM

set -e

# Configuration
MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
GPUS="2,7"  # Use GPUs 6 and 7 for tensor parallelism
PORT=8003
LOG_FILE="tmp/vllm_Qwen3-30B-A3B-Instruct-2507_${PORT}.log"

# Create log directory
mkdir -p tmp

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate verl  # Adjust to your vLLM environment

export PYTHONPATH=./

echo "=================================================================================="
echo "Starting vLLM server for Qwen3-30B-A3B-Instruct-2507"
echo "=================================================================================="
echo "Model: $MODEL_PATH"
echo "GPUs: $GPUS (tensor parallelism)"
echo "Port: $PORT"
echo "Log: $LOG_FILE"
echo ""

# Start vLLM server in background on GPUs 6 and 7
echo "Starting vLLM server on GPUs $GPUS with tensor parallelism..."
(
    export CUDA_VISIBLE_DEVICES=$GPUS
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --host "0.0.0.0" \
        --gpu-memory-utilization 0.6 \
        --tensor-parallel-size 2 \
        > "$LOG_FILE" 2>&1
) &

VLLM_PID=$!
echo "vLLM server started (PID: $VLLM_PID) on port $PORT using GPUs $GPUS"
echo ""

# Wait for server to be ready
echo "Waiting for server to be ready..."
max_attempts=120  # 4 minutes timeout for large model
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "✓ Server is ready!"
        echo ""
        echo "=================================================================================="
        echo "Server is running!"
        echo "=================================================================================="
        echo "Endpoint: http://localhost:$PORT/v1/chat/completions"
        echo "API Key: lgms_sk_live"
        echo ""
        echo "To stop the server, run: kill $VLLM_PID"
        echo "To view logs: tail -f $LOG_FILE"
        echo "=================================================================================="
        break
    fi
    sleep 2
    attempt=$((attempt + 1))
    if [ $((attempt % 10)) -eq 0 ]; then
        echo "Still waiting... (attempt $attempt/$max_attempts)"
    fi
done

if [ $attempt -eq $max_attempts ]; then
    echo "Error: Server failed to start within timeout"
    echo "Check logs: $LOG_FILE"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Keep script running
wait $VLLM_PID
