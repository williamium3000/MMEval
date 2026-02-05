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
SAVE_DIR=work_dirs/vg/ablation_v18_nocontext_v2
RUN_FILE=examiner/ablation/dyna_conv_v18resume_nocontext_v2.py
LOG_DIR=work_dirs/logs_ablation_v18_nocontext_v2

# Create log directory
mkdir -p ${LOG_DIR}

# Initialize conda
eval "$(conda shell.bash hook)"

echo "Starting parallel inference jobs..."
echo "Logs will be saved to: ${LOG_DIR}"

# ===== Env: qwenvl - Qwen2.5-VL Series =====
(
  OUTFILE=$SAVE_DIR/Qwen2.5-VL-3B-Instruct.json
  LOGFILE=${LOG_DIR}/Qwen2.5-VL-3B-Instruct.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping Qwen2.5-VL-3B-Instruct - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/qwenvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path Qwen/Qwen2.5-VL-3B-Instruct --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/Qwen2.5-VL-3B-Instruct.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/Qwen2.5-VL-7B-Instruct.json
  LOGFILE=${LOG_DIR}/Qwen2.5-VL-7B-Instruct.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping Qwen2.5-VL-7B-Instruct - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/qwenvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path Qwen/Qwen2.5-VL-7B-Instruct --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/Qwen2.5-VL-7B-Instruct.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/Qwen2.5-VL-72B-Instruct.json
  LOGFILE=${LOG_DIR}/Qwen2.5-VL-72B-Instruct.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping Qwen2.5-VL-72B-Instruct - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/qwenvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
    python $RUN_FILE --dataset vg --model_path Qwen/Qwen2.5-VL-72B-Instruct --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/Qwen2.5-VL-72B-Instruct.log 2>&1 &

# ===== Env: qwenvl - Qwen3-VL Series =====
(
  OUTFILE=$SAVE_DIR/Qwen3-VL-2B-Instruct.json
  LOGFILE=${LOG_DIR}/Qwen3-VL-2B-Instruct.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping Qwen3-VL-2B-Instruct - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/qwenvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path Qwen/Qwen3-VL-2B-Instruct --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/Qwen3-VL-2B-Instruct.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/Qwen3-VL-8B-Instruct.json
  LOGFILE=${LOG_DIR}/Qwen3-VL-8B-Instruct.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping Qwen3-VL-8B-Instruct - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/qwenvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path Qwen/Qwen3-VL-8B-Instruct --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/Qwen3-VL-8B-Instruct.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/Qwen3-VL-32B-Instruct.json
  LOGFILE=${LOG_DIR}/Qwen3-VL-32B-Instruct.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping Qwen3-VL-32B-Instruct - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/qwenvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
    python $RUN_FILE --dataset vg --model_path Qwen/Qwen3-VL-32B-Instruct --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/Qwen3-VL-32B-Instruct.log 2>&1 &

# ===== Env: llava - LLaVA Series =====
(
  OUTFILE=$SAVE_DIR/llava-1.5-7b-hf.json
  LOGFILE=${LOG_DIR}/llava-1.5-7b-hf.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping llava-1.5-7b-hf - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/llava
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path llava-hf/llava-1.5-7b-hf --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/llava-1.5-7b-hf.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/llava-1.5-13b-hf.json
  LOGFILE=${LOG_DIR}/llava-1.5-13b-hf.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping llava-1.5-13b-hf - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/llava
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path llava-hf/llava-1.5-13b-hf --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/llava-1.5-13b-hf.log 2>&1 &


# ===== Env: internvl - InternVL2 Series =====
(
  OUTFILE=$SAVE_DIR/InternVL2-2B.json
  LOGFILE=${LOG_DIR}/InternVL2-2B.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping InternVL2-2B - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/internvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2-2B --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/InternVL2-2B.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/InternVL2-8B.json
  LOGFILE=${LOG_DIR}/InternVL2-8B.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping InternVL2-8B - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/internvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2-8B --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/InternVL2-8B.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/InternVL2-26B.json
  LOGFILE=${LOG_DIR}/InternVL2-26B.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping InternVL2-26B - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/internvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
    python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2-26B --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/InternVL2-26B.log 2>&1 &

# ===== Env: internvl - InternVL2.5 Series =====
(
  OUTFILE=$SAVE_DIR/InternVL2_5-2B.json
  LOGFILE=${LOG_DIR}/InternVL2_5-2B.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping InternVL2_5-2B - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/internvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2_5-2B --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/InternVL2_5-2B.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/InternVL2_5-8B.json
  LOGFILE=${LOG_DIR}/InternVL2_5-8B.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping InternVL2_5-8B - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/internvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2_5-8B --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/InternVL2_5-8B.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/InternVL2_5-38B.json
  LOGFILE=${LOG_DIR}/InternVL2_5-38B.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping InternVL2_5-38B - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/internvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
    python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2_5-38B --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/InternVL2_5-38B.log 2>&1 &

# ===== Env: internvl - InternVL3 Series =====
(
  OUTFILE=$SAVE_DIR/InternVL3-2B-Instruct.json
  LOGFILE=${LOG_DIR}/InternVL3-2B-Instruct.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping InternVL3-2B-Instruct - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/internvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL3-2B-Instruct --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/InternVL3-2B-Instruct.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/InternVL3-8B-Instruct.json
  LOGFILE=${LOG_DIR}/InternVL3-8B-Instruct.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping InternVL3-8B-Instruct - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/internvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL3-8B-Instruct --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/InternVL3-8B-Instruct.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/InternVL3-38B-Instruct.json
  LOGFILE=${LOG_DIR}/InternVL3-38B-Instruct.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping InternVL3-38B-Instruct - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/internvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
    python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL3-38B-Instruct --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/InternVL3-38B-Instruct.log 2>&1 &

# ===== Env: flan_t5 - BLIP2 Series =====
(
  OUTFILE=$SAVE_DIR/blip2-flan-t5-xl.json
  LOGFILE=${LOG_DIR}/blip2-flan-t5-xl.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping blip2-flan-t5-xl - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/qwenvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path Salesforce/blip2-flan-t5-xl --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/blip2-flan-t5-xl.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/blip2-flan-t5-xxl.json
  LOGFILE=${LOG_DIR}/blip2-flan-t5-xxl.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping blip2-flan-t5-xxl - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/qwenvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path Salesforce/blip2-flan-t5-xxl --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/blip2-flan-t5-xxl.log 2>&1 &

# ===== Env: instructblip - InstructBLIP Series =====
(
  OUTFILE=$SAVE_DIR/instructblip-vicuna-7b.json
  LOGFILE=${LOG_DIR}/instructblip-vicuna-7b.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping instructblip-vicuna-7b - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/qwenvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path Salesforce/instructblip-vicuna-7b --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/instructblip-vicuna-7b.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/instructblip-vicuna-13b.json
  LOGFILE=${LOG_DIR}/instructblip-vicuna-13b.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping instructblip-vicuna-13b - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/qwenvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path Salesforce/instructblip-vicuna-13b --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/instructblip-vicuna-13b.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/instructblip-flan-t5-xxl.json
  LOGFILE=${LOG_DIR}/instructblip-flan-t5-xxl.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping instructblip-flan-t5-xxl - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/qwenvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path Salesforce/instructblip-flan-t5-xxl --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/instructblip-flan-t5-xxl.log 2>&1 &

# ===== Env: gemma3 - Gemma3 Series =====
(
  OUTFILE=$SAVE_DIR/gemma-3-4b-it.json
  LOGFILE=${LOG_DIR}/gemma-3-4b-it.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping gemma-3-4b-it - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/gemma3
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path google/gemma-3-4b-it --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/gemma-3-4b-it.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/gemma-3-12b-it.json
  LOGFILE=${LOG_DIR}/gemma-3-12b-it.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping gemma-3-12b-it - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/gemma3
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path google/gemma-3-12b-it --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/gemma-3-12b-it.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/gemma-3-27b-it.json
  LOGFILE=${LOG_DIR}/gemma-3-27b-it.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping gemma-3-27b-it - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/gemma3
    export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
    python $RUN_FILE --dataset vg --model_path google/gemma-3-27b-it --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/gemma-3-27b-it.log 2>&1 &



# ===== Env: phi3v - Phi3.5 Series =====
(
  OUTFILE=$SAVE_DIR/Phi-3.5-vision-instruct.json
  LOGFILE=${LOG_DIR}/Phi-3.5-vision-instruct.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping Phi-3.5-vision-instruct - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/phi4
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path microsoft/Phi-3.5-vision-instruct --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/Phi-3.5-vision-instruct.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/Phi-4-multimodal-instruct.json
  LOGFILE=${LOG_DIR}/Phi-4-multimodal-instruct.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping Phi-4-multimodal-instruct - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/phi4
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path microsoft/Phi-4-multimodal-instruct --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/Phi-4-multimodal-instruct.log 2>&1 &

# ===== Env: gemma3 - PaliGemma 2 (10B, 28B) =====
(
  OUTFILE=$SAVE_DIR/paligemma2-10b-mix-224.json
  LOGFILE=${LOG_DIR}/paligemma2-10b-mix-224.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping paligemma2-10b-mix-224 - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/gemma3
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path google/paligemma2-10b-mix-224 --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/paligemma2-10b-mix-224.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/paligemma2-28b-mix-224.json
  LOGFILE=${LOG_DIR}/paligemma2-28b-mix-224.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping paligemma2-28b-mix-224 - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/gemma3
    export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
    python $RUN_FILE --dataset vg --model_path google/paligemma2-28b-mix-224 --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/paligemma2-28b-mix-224.log 2>&1 &

# ===== Env: ovis2 - Ovis2 (1B, 2B, 4B, 16B, 34B) =====
(
  OUTFILE=$SAVE_DIR/Ovis2-1B.json
  LOGFILE=${LOG_DIR}/Ovis2-1B.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping Ovis2-1B - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/ovis2
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path AIDC-AI/Ovis2-1B --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/Ovis2-1B.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/Ovis2-2B.json
  LOGFILE=${LOG_DIR}/Ovis2-2B.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping Ovis2-2B - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/ovis2
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path AIDC-AI/Ovis2-2B --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/Ovis2-2B.log 2>&1 &

(
  OUTFILE=$SAVE_DIR/Ovis2-34B.json
  LOGFILE=${LOG_DIR}/Ovis2-34B.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping Ovis2-34B - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/ovis2
    export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
    python $RUN_FILE --dataset vg --model_path AIDC-AI/Ovis2-34B --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/Ovis2-34B.log 2>&1 &

# ===== Env: llava - LLaVA-RLHF-7B =====
(
  OUTFILE=$SAVE_DIR/LLaVA-RLHF-7b-v1.5-224.json
  LOGFILE=${LOG_DIR}/LLaVA-RLHF-7b-v1.5-224.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping LLaVA-RLHF-7b-v1.5-224 - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/llava
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path zhiqings/LLaVA-RLHF-7b-v1.5-224 --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/LLaVA-RLHF-7b-v1.5-224.log 2>&1 &

# ===== Env: phi4 - Idefics2 base and chatty variants =====
(
  OUTFILE=$SAVE_DIR/idefics2-8b-lpoi-list5-10k.json
  LOGFILE=${LOG_DIR}/idefics2-8b-lpoi-list5-10k.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping idefics2-8b-lpoi-list5-10k - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/phi4
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path data/checkpoints/idefics2-8b-lpoi-list5-10k/final --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/idefics2-8b-lpoi-list5-10k.log 2>&1 &

# Wait for all background jobs to complete
wait

echo "All parallel inference jobs completed!"
echo "Check logs in: ${LOG_DIR}"