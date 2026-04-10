#!/bin/bash
# vLLM server for MMHAL local grading: 2 GPUs (4,5) at 30% utilization
# Then run: bash scripts/graders/mmhal.sh --local work_dirs/vg/caption

set -e

# Configuration
MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
GPUS="0,1"
PORT=8004
GPU_UTIL=0.5
TP_SIZE=2
LOG_FILE="tmp/vllm_Qwen3-30B-A3B-Instruct-2507_${PORT}_gpu45_util30.log"


# Create log directory
mkdir -p tmp

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate verl

export PYTHONPATH=./

echo "=================================================================================="
echo "Starting vLLM server (MMHAL local)"
echo "=================================================================================="
echo "Model: $MODEL_PATH"
echo "GPUs: $GPUS (tensor-parallel-size $TP_SIZE)"
echo "GPU memory utilization: ${GPU_UTIL}"
echo "Port: $PORT"
echo "Log: $LOG_FILE"
echo ""

echo "Starting vLLM server (VLLM_USE_SYMM_MEM=0)..."
(
    export CUDA_VISIBLE_DEVICES=$GPUS
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --host "0.0.0.0" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --tensor-parallel-size "$TP_SIZE" \
        > "$LOG_FILE" 2>&1
) &

VLLM_PID=$!
echo "vLLM server started (PID: $VLLM_PID) on port $PORT"
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
        echo ""
        echo "Run MMHAL with: bash scripts/graders/mmhal.sh --local work_dirs/vg/caption"
        echo ""
        echo "To stop: kill $VLLM_PID"
        echo "Logs:    tail -f $LOG_FILE"
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

wait $VLLM_PID
