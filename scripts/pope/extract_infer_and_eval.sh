#!/bin/bash
# 3-step pipeline:
# 1. Qwen extracts yes/no questions and response-based answers (port 8004)
# 2. VLM answers the questions (auto-host VLM, need GPUs)
# 3. Verify correctness of both to compare accuracy

set -e

export PYTHONPATH=./
export PYTHONPATH=$PYTHONPATH:/raid/william/project/context-eval-mllm/infer/LLaVA/llava
export PYTHONPATH="/raid/william/project/context-eval-mllm/infer/LLaVA:${PYTHONPATH:-}"

eval "$(conda shell.bash hook)"
conda activate verl

export PYTHONPATH=./

# Parse arguments
INPUT_DIR=""
VLM_MODEL_PATH=""
VLM_MODEL_BASE=""
VLM_GPUS=""
VLM_PORT=8005
QWEN_PORT=8004
SAMPLE_NUM=""
START_IDX=0
BATCH_SIZE=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --vlm-model-path)
            VLM_MODEL_PATH="$2"
            shift 2
            ;;
        --vlm-model-base)
            VLM_MODEL_BASE="$2"
            shift 2
            ;;
        --vlm-gpus)
            VLM_GPUS="$2"
            shift 2
            ;;
        --vlm-port)
            VLM_PORT="$2"
            shift 2
            ;;
        --qwen-port)
            QWEN_PORT="$2"
            shift 2
            ;;
        --sample-num)
            SAMPLE_NUM="$2"
            shift 2
            ;;
        --start-idx)
            START_IDX="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set defaults
INPUT_DIR="${INPUT_DIR:-work_dirs/vg/final_run_v18_gpt4o_completed}"

if [ -z "$VLM_MODEL_PATH" ]; then
    echo "Error: --vlm-model-path is required (use comma to run multiple models in parallel)"
    echo "Example: --vlm-model-path llava-hf/llava-1.5-7b-hf"
    echo "Example: --vlm-model-path llava-hf/llava-1.5-7b-hf,InternVL/InternVL2_5-2B"
    exit 1
fi

# Support multiple models (comma-separated); GPUs: "7,8|9,10" = per-model, else same for all
IFS=',' read -ra VLM_MODELS <<< "$VLM_MODEL_PATH"
NUM_MODELS=${#VLM_MODELS[@]}
if [ -n "$VLM_GPUS" ] && [[ "$VLM_GPUS" == *"|"* ]]; then
    IFS='|' read -ra VLM_GPUS_PER_MODEL <<< "$VLM_GPUS"
else
    VLM_GPUS_DEFAULT="${VLM_GPUS:-2,3}"
    VLM_GPUS_PER_MODEL=()
    for ((i=0;i<NUM_MODELS;i++)); do VLM_GPUS_PER_MODEL+=("$VLM_GPUS_DEFAULT"); done
fi

echo "=================================================================================="
echo "3-Step Pipeline: Extract, Infer, and Evaluate (${NUM_MODELS} model(s) in parallel)"
echo "=================================================================================="
echo "Input directory: $INPUT_DIR"
echo "VLM model(s): ${VLM_MODELS[*]}"
echo "VLM port (base): $VLM_PORT"
echo "Qwen port: $QWEN_PORT"
echo ""

# Per-model: pope file, basename, and mode (verify_only if _pope_converted exists; else full = convert then verify)
POPE_FILES=()
VLM_MODEL_NAMES=()
VERIFY_ONLY=()
for ((i=0;i<NUM_MODELS;i++)); do
    mp="${VLM_MODELS[$i]}"
    name=$(basename "$mp")
    VLM_MODEL_NAMES+=("$name")
    pope_match="$INPUT_DIR/${name}_pope_converted.json"
    model_json="$INPUT_DIR/${name}.json"
    if [ -f "$pope_match" ]; then
        POPE_FILES+=("$pope_match")
        VERIFY_ONLY+=("1")
        echo "[$name] Found _pope_converted.json -> verify_only (VLM + Qwen GT only, no full transcript)"
    elif [ -f "$model_json" ]; then
        echo "[$name] No _pope_converted; found ${name}.json -> full pipeline (question generation then verify)"
        echo "  Creating _pope_converted from ${name}.json..."
        conv_cmd="python examiner/DSG_v2.py --conv_script $model_json --outfile $pope_match --extract_from_transcript --start_idx $START_IDX --local"
        [ -n "$SAMPLE_NUM" ] && conv_cmd="$conv_cmd --sample_num $SAMPLE_NUM"
        eval "$conv_cmd" || { echo "Error: DSG_v2 conversion failed for $name"; exit 1; }
        if [ -f "$pope_match" ]; then
            python utils/extract_dsg_qa_questions.py "$pope_match" 2>/dev/null || true
        fi
        POPE_FILES+=("$pope_match")
        VERIFY_ONLY+=("0")
    else
        echo "Error: No file found for VLM $name (need $pope_match or $model_json)"
        exit 1
    fi
done
echo "Found ${NUM_MODELS} file(s) to process"
echo ""

echo "=================================================================================="
echo "Step 1: Qwen (port $QWEN_PORT) for GT / transcript extraction"
echo "=================================================================================="
echo "Using Qwen on port $QWEN_PORT"
echo ""

if ! curl -s "http://localhost:$QWEN_PORT/health" > /dev/null 2>&1; then
    echo "Error: Qwen server not running on port $QWEN_PORT"
    echo "Please start Qwen server first: bash scripts/host_qwen3_30b_gpu01.sh"
    exit 1
fi

echo "=================================================================================="
echo "Step 2: Auto-host VLM(s) and get fresh answers (all in parallel)"
echo "=================================================================================="

vlm_ports=()
vlm_pids=()
vlm_we_own=()
for ((i=0;i<NUM_MODELS;i++)); do vlm_we_own+=("0"); vlm_pids+=("0"); done

for ((i=0;i<NUM_MODELS;i++)); do
    mp="${VLM_MODELS[$i]}"
    name="${VLM_MODEL_NAMES[$i]}"
    gpus="${VLM_GPUS_PER_MODEL[$i]}"
    preferred_port=$((VLM_PORT + i))
    try_port=$preferred_port
    while true; do
        if curl -s "http://localhost:$try_port/health" > /dev/null 2>&1; then
            models_json=$(curl -s "http://localhost:$try_port/v1/models" 2>/dev/null || true)
            if echo "$models_json" | grep -q "$name"; then
                vlm_ports+=("$try_port")
                echo "[$name] Already running on port $try_port - reusing."
                break
            fi
            try_port=$((try_port + 1))
        else
            vlm_ports+=("$try_port")
            LOG_FILE="tmp/vllm_${name}_${try_port}.log"
            VLLM_EXTRA=()
            case "$mp" in
                *llava*|*Llava*|*LLaVA*)
                    VLLM_EXTRA=(--trust-remote-code)
                    ;;
            esac
            (
                export CUDA_VISIBLE_DEVICES=$gpus
                python -m vllm.entrypoints.openai.api_server \
                    --model "$mp" \
                    --port "$try_port" \
                    --host "0.0.0.0" \
                    --gpu-memory-utilization 0.6 \
                    --tensor-parallel-size $(echo "$gpus" | tr ',' '\n' | wc -l) \
                    "${VLLM_EXTRA[@]}" \
                    > "$LOG_FILE" 2>&1
            ) &
            vlm_pids[$i]=$!
            vlm_we_own[$i]=1
            echo "[$name] Started vLLM (PID: ${vlm_pids[$i]}) on port $try_port"
            break
        fi
    done
done

echo "Waiting for all VLM server(s) to be ready..."
for ((i=0;i<NUM_MODELS;i++)); do
    name="${VLM_MODEL_NAMES[$i]}"
    port="${vlm_ports[$i]}"
    max_attempts=120
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "[$name] Ready on port $port"
            break
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    if [ $attempt -eq $max_attempts ]; then
        echo "Error: VLM $name failed to become ready on port $port"
        for ((j=0;j<NUM_MODELS;j++)); do
            [ "${vlm_we_own[$j]}" -eq 1 ] && kill "${vlm_pids[$j]}" 2>/dev/null || true
        done
        exit 1
    fi
done

echo ""
echo "=================================================================================="
echo "Step 3: Extract answers, infer with VLM, and verify (all models in parallel)"
echo "=================================================================================="

step3_pids=()
for ((i=0;i<NUM_MODELS;i++)); do
    pope_file="${POPE_FILES[$i]}"
    filename=$(basename "$pope_file")
    outfile="${pope_file%_pope_converted.json}_with_both_answers.json"
    port="${vlm_ports[$i]}"
    mp="${VLM_MODELS[$i]}"
    name="${VLM_MODEL_NAMES[$i]}"

    use_verify_only="${VERIFY_ONLY[$i]}"
    resume_flag=""
    if [ -f "$outfile" ]; then
        resume_flag="--resume"
        echo "[$(date +'%H:%M:%S')] Resuming: $outfile exists, will skip already-processed samples"
    fi
    echo "[$(date +'%H:%M:%S')] Starting: $filename (port $port) mode=$([ "$use_verify_only" = "1" ] && echo verify_only || echo full)"
    (
        cmd="python examiner/DSG_v4.py --input_file \"$pope_file\" --outfile \"$outfile\""
        cmd="$cmd --vlm_api_url http://localhost:$port/v1 --vlm_api_model \"$mp\""
        cmd="$cmd --qwen_port $QWEN_PORT --batch_size $BATCH_SIZE --start_idx $START_IDX --verify"
        [ -n "$resume_flag" ] && cmd="$cmd $resume_flag"
        [ "$use_verify_only" = "1" ] && cmd="$cmd --verify_only"
        [ -n "$SAMPLE_NUM" ] && cmd="$cmd --sample_num $SAMPLE_NUM"
        if eval "$cmd"; then
            echo "[$(date +'%H:%M:%S')] [$name] ✅ Completed: $filename"
        else
            echo "[$(date +'%H:%M:%S')] [$name] ❌ Failed: $filename"
        fi
    ) &
    step3_pids+=($!)
done
processed=0
for ((i=0;i<NUM_MODELS;i++)); do
    wait "${step3_pids[$i]}" && ((processed++)) || true
done

echo ""
echo "Stopping VLM server(s) we started..."
for ((i=0;i<NUM_MODELS;i++)); do
    if [ "${vlm_we_own[$i]}" -eq 1 ]; then
        echo "  Stopping ${VLM_MODEL_NAMES[$i]} (PID: ${vlm_pids[$i]})"
        kill "${vlm_pids[$i]}" 2>/dev/null || true
        wait "${vlm_pids[$i]}" 2>/dev/null || true
    else
        echo "  Leaving ${VLM_MODEL_NAMES[$i]} on port ${vlm_ports[$i]} (was already running)."
    fi
done

echo ""
echo "=================================================================================="
echo "Summary:"
echo "  Processed: $processed / $NUM_MODELS"
echo "=================================================================================="
echo ""
echo "⚠️  For multiple models use different GPUs: --vlm-gpus \"7|8\" or \"7,8|9,10\""
