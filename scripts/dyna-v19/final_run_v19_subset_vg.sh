#!/bin/bash

# v19 VG subset: InternVL3-8B-Instruct, Qwen2.5-VL-7B-Instruct, opera-llava-1.5
# Mirrors final_run_all_parallel_small_vg_v19.sh, plus opera (from v20 template).

if [ -z "$1" ]; then
    echo "Usage: $0 <cuda_devices>"
    echo "Example: $0 0,1,2,3"
    exit 1
fi

IFS=',' read -ra CUDA_DEVICES <<< "$1"
NUM_GPUS=${#CUDA_DEVICES[@]}

echo "Available GPUs: ${CUDA_DEVICES[@]}"
echo "Total GPUs: $NUM_GPUS"

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
SAVE_DIR=work_dirs/vg/v19
RUN_FILE=examiner/dyna_conv_v19.py
LOG_DIR=work_dirs/logs_v19

mkdir -p ${SAVE_DIR} ${LOG_DIR}

eval "$(conda shell.bash hook)"

echo "Starting v19 subset VG inference (3 models)..."
echo "  save_dir: ${SAVE_DIR}"
echo "  log_dir:  ${LOG_DIR}"

# ===== InternVL3-8B-Instruct =====
(
  OUTFILE=$SAVE_DIR/InternVL3-8B-Instruct.json
  LOGFILE=${LOG_DIR}/InternVL3-8B-Instruct.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping InternVL3-8B-Instruct - output exists and no errors: $OUTFILE"
  else
    conda activate work_dirs/envs/internvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL3-8B-Instruct \
      --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/InternVL3-8B-Instruct.log 2>&1 &

# ===== Qwen2.5-VL-7B-Instruct =====
(
  OUTFILE=$SAVE_DIR/Qwen2.5-VL-7B-Instruct.json
  LOGFILE=${LOG_DIR}/Qwen2.5-VL-7B-Instruct.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping Qwen2.5-VL-7B-Instruct - output exists and no errors: $OUTFILE"
  else
    conda activate work_dirs/envs/qwenvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path Qwen/Qwen2.5-VL-7B-Instruct \
      --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/Qwen2.5-VL-7B-Instruct.log 2>&1 &

# ===== opera-llava-1.5 =====
(
  OUTFILE=$SAVE_DIR/opera-llava-1.5.json
  LOGFILE=${LOG_DIR}/opera-llava-1.5.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping opera-llava-1.5 - output exists and no errors: $OUTFILE"
  else
    conda activate opera
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg \
      --model_path /raid/william/project/context-eval-mllm/data/checkpoints/opera/llava-1.5 \
      --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/opera-llava-1.5.log 2>&1 &

wait

echo "All v19 subset VG jobs completed!"
echo "Outputs in : ${SAVE_DIR}"
echo "Logs in    : ${LOG_DIR}"
