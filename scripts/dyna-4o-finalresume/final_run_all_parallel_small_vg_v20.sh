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
SAVE_DIR=work_dirs/vg/final_run_v20_gpt4o_small
RUN_FILE=examiner/dyna_conv_v20.py
LOG_DIR=work_dirs/logs_parallel_v20_small

# Create output and log directories
mkdir -p ${SAVE_DIR}
mkdir -p ${LOG_DIR}

# Initialize conda
eval "$(conda shell.bash hook)"

echo "Starting parallel inference jobs..."
echo "Logs will be saved to: ${LOG_DIR}"

# ===== Env: llava - LLaVA-7B =====
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

# ===== Env: internvl - InternVL2-8B =====
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

# ===== Env: internvl - InternVL2.5-8B =====
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

# ===== Env: internvl - InternVL3-8B =====
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

# ===== Env: qwenvl - Qwen2.5-VL-7B =====
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

# ===== Env: gemma3 - gemma-3-12b-it =====
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

# ===== Env: opera - opera-llava-1.5 =====
(
  OUTFILE=$SAVE_DIR/opera-llava-1.5.json
  LOGFILE=${LOG_DIR}/opera-llava-1.5.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping opera-llava-1.5 - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate opera
    export CUDA_VISIBLE_DEVICES=0
    python $RUN_FILE --dataset vg --model_path /raid/william/project/context-eval-mllm/data/checkpoints/opera/llava-1.5 --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/opera-llava-1.5.log 2>&1 &

# Wait for all background jobs to complete
wait

echo "All parallel inference jobs completed!"
echo "Check logs in: ${LOG_DIR}"
