#!/bin/bash

# Check if CUDA devices argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <cuda_devices>"
    echo "Example: $0 0,1,2,3,4,5,6,7"
    exit 1
fi

# Parse CUDA devices into array
IFS=',' read -ra CUDA_DEVICES <<< "$1"
NUM_GPUS=${#CUDA_DEVICES[@]}

echo "Available GPUs: ${CUDA_DEVICES[@]}"
echo "Total GPUs: $NUM_GPUS"

# Function to randomly select N GPUs
select_gpus() {
    local num_needed=$1
    local selected=()
    local available=("${CUDA_DEVICES[@]}")
    
    for ((i=0; i<num_needed; i++)); do
        local idx=$((RANDOM % ${#available[@]}))
        selected+=("${available[$idx]}")
        available=("${available[@]:0:$idx}" "${available[@]:$((idx+1))}")
    done
    
    IFS=','
    echo "${selected[*]}"
}

export PYTHONPATH=$PYTHONPATH:./
export PYTHONPATH=$PYTHONPATH:/raid/william/project/context-eval-mllm/infer/LLaVA/llava
export PYTHONPATH="/raid/william/project/context-eval-mllm/infer/LLaVA:${PYTHONPATH:-}"

NUM_SAMPLES=100
SAVE_DIR=work_dirs/vg/final_run_v18_gpt4o
RUN_FILE=examiner/dyna_conv_v18resume.py
LOG_DIR=work_dirs/logs_parallel

# Create log directory
mkdir -p ${LOG_DIR}

# Initialize conda
eval "$(conda shell.bash hook)"

echo "Starting parallel inference jobs..."
echo "Logs will be saved to: ${LOG_DIR}"

# ===== Env: qwenvl - Qwen2.5-VL Series =====
(
  conda activate work_dirs/envs/qwenvl
  export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
  python $RUN_FILE --dataset vg --model_path Qwen/Qwen2.5-VL-72B-Instruct --outfile $SAVE_DIR/Qwen2.5-VL-72B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Qwen2.5-VL-72B-Instruct.log 2>&1 &

# ===== Env: qwenvl - Qwen3-VL Series =====
(
  conda activate work_dirs/envs/qwenvl
  export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
  python $RUN_FILE --dataset vg --model_path Qwen/Qwen3-VL-2B-Instruct --outfile $SAVE_DIR/Qwen3-VL-2B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Qwen3-VL-2B-Instruct.log 2>&1 &

(
  conda activate work_dirs/envs/qwenvl
  export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
  python $RUN_FILE --dataset vg --model_path Qwen/Qwen3-VL-8B-Instruct --outfile $SAVE_DIR/Qwen3-VL-8B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Qwen3-VL-8B-Instruct.log 2>&1 &

(
  conda activate work_dirs/envs/qwenvl
  export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
  python $RUN_FILE --dataset vg --model_path Qwen/Qwen3-VL-32B-Instruct --outfile $SAVE_DIR/Qwen3-VL-32B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Qwen3-VL-32B-Instruct.log 2>&1 &

# ===== Env: llava - LLaVA Series =====
# ===== Env: llava_next - LLaVA Series =====
(
  conda activate work_dirs/envs/llava
  export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
  python $RUN_FILE --dataset vg --model_path llava-hf/llava-v1.6-vicuna-7b-hf --outfile $SAVE_DIR/llava-v1.6-vicuna-7b-hf.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/llava-v1.6-vicuna-7b-hf.log 2>&1 &

(
  conda activate work_dirs/envs/llava
  export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
  python $RUN_FILE --dataset vg --model_path llava-hf/llava-v1.6-vicuna-13b-hf --outfile $SAVE_DIR/llava-v1.6-vicuna-13b-hf.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/llava-v1.6-vicuna-13b-hf.log 2>&1 &


# ===== Env: internvl - InternVL2 Series =====
(
  conda activate work_dirs/envs/internvl
  export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2-2B --outfile $SAVE_DIR/InternVL2-2B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2-2B.log 2>&1 &

(
  conda activate work_dirs/envs/internvl
  export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2-26B --outfile $SAVE_DIR/InternVL2-26B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2-26B.log 2>&1 &

# ===== Env: internvl - InternVL2.5 Series =====
(
  conda activate work_dirs/envs/internvl
  export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2_5-2B --outfile $SAVE_DIR/InternVL2_5-2B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2_5-2B.log 2>&1 &

(
  conda activate work_dirs/envs/internvl
  export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2_5-8B --outfile $SAVE_DIR/InternVL2_5-8B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2_5-8B.log 2>&1 &

(
  conda activate work_dirs/envs/internvl
  export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2_5-38B --outfile $SAVE_DIR/InternVL2_5-38B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2_5-38B.log 2>&1 &

# ===== Env: internvl - InternVL3 Series =====
(
  conda activate work_dirs/envs/internvl
  export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL3-2B-Instruct --outfile $SAVE_DIR/InternVL3-2B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL3-2B-Instruct.log 2>&1 &


(
  conda activate work_dirs/envs/internvl
  export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL3-38B-Instruct --outfile $SAVE_DIR/InternVL3-38B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL3-38B-Instruct.log 2>&1 &

# ===== Env: flan_t5 - BLIP2 Series =====

# ===== Env: instructblip - InstructBLIP Series =====

# ===== Env: gemma3 - Gemma3 Series =====
(
  conda activate work_dirs/envs/gemma3
  export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
  python $RUN_FILE --dataset vg --model_path google/gemma-3-4b-it --outfile $SAVE_DIR/gemma-3-4b-it.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/gemma-3-4b-it.log 2>&1 &

(
  conda activate work_dirs/envs/gemma3
  export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
  python $RUN_FILE --dataset vg --model_path google/gemma-3-12b-it --outfile $SAVE_DIR/gemma-3-12b-it.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/gemma-3-12b-it.log 2>&1 &

(
  conda activate work_dirs/envs/gemma3
  export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
  python $RUN_FILE --dataset vg --model_path google/gemma-3-27b-it --outfile $SAVE_DIR/gemma-3-27b-it.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/gemma-3-27b-it.log 2>&1 &



# ===== Env: phi3v - Phi3.5 Series =====
(
  conda activate work_dirs/envs/phi4
  export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
  python $RUN_FILE --dataset vg --model_path microsoft/Phi-3.5-vision-instruct --outfile $SAVE_DIR/Phi-3.5-vision-instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Phi-3.5-vision-instruct.log 2>&1 &

# ===== Env: gemma3 - PaliGemma 2 (10B, 28B) =====
(
  conda activate work_dirs/envs/gemma3
  export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
  python $RUN_FILE --dataset vg --model_path google/paligemma2-28b-mix-224 --outfile $SAVE_DIR/paligemma2-28b-mix-224.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/paligemma2-28b-mix-224.log 2>&1 &

# ===== Env: ovis2 - Ovis2 (1B, 2B, 4B, 16B, 34B) =====
(
  conda activate work_dirs/envs/ovis2
  export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
  python $RUN_FILE --dataset vg --model_path AIDC-AI/Ovis2-1B --outfile $SAVE_DIR/Ovis2-1B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Ovis2-1B.log 2>&1 &

(
  conda activate work_dirs/envs/ovis2
  export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
  python $RUN_FILE --dataset vg --model_path AIDC-AI/Ovis2-34B --outfile $SAVE_DIR/Ovis2-34B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Ovis2-34B.log 2>&1 &

# ===== Env: llava - LLaVA-RLHF-7B =====
(
  conda activate work_dirs/envs/llava
  export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
  python $RUN_FILE --dataset vg --model_path zhiqings/LLaVA-RLHF-7b-v1.5-224 --outfile $SAVE_DIR/LLaVA-RLHF-7b-v1.5-224.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/LLaVA-RLHF-7b-v1.5-224.log 2>&1 &

# ===== Env: phi4 - Idefics2 base and chatty variants =====
(
  conda activate work_dirs/envs/phi4
  export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
  python $RUN_FILE --dataset vg --model_path data/checkpoints/idefics2-8b-lpoi-list5-10k/final --outfile $SAVE_DIR/idefics2-8b-lpoi-list5-10k.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/idefics2-8b-lpoi-list5-10k.log 2>&1 &

# Wait for all background jobs to complete
wait

echo "All parallel inference jobs completed!"
echo "Check logs in: ${LOG_DIR}"