#!/bin/bash

# Caption (examiner/caption.py) — SVG dataset, 500 samples.
# Models: LLaVA-7B, InternVL2-8B, InternVL2.5-8B, InternVL3-8B,
#         Qwen2.5-VL-7B, gemma-3-12b-it, opera-1.5

if [ -z "$1" ]; then
    echo "Usage: $0 <cuda_devices>"
    echo "Example: $0 0,1,2,3,4,5,6,7"
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

export PYTHONPATH=$PYTHONPATH:./:infer:grader/easydetect
export PYTHONPATH=$PYTHONPATH:/raid/william/project/context-eval-mllm/infer/LLaVA/llava
export PYTHONPATH="/raid/william/project/context-eval-mllm/infer/LLaVA:${PYTHONPATH:-}"

DATASET=svg
NUM_SAMPLES=500
SAVE_DIR=work_dirs/svg/caption
LOG_DIR=work_dirs/logs_caption_svg

mkdir -p ${SAVE_DIR} ${LOG_DIR}

eval "$(conda shell.bash hook)"

echo "Starting caption SVG parallel inference..."
echo "  dataset     : ${DATASET}"
echo "  num_samples : ${NUM_SAMPLES}"
echo "  save_dir    : ${SAVE_DIR}"
echo "  log_dir     : ${LOG_DIR}"

# ===== Env: llava - LLaVA-7B =====
(
  OUTFILE=$SAVE_DIR/llava-1.5-7b-hf.json
  LOGFILE=${LOG_DIR}/llava-1.5-7b-hf.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 50 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping llava-1.5-7b-hf - output exists and no errors: $OUTFILE"
  else
    conda activate work_dirs/envs/llava
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python examiner/caption.py --dataset $DATASET --model_path llava-hf/llava-1.5-7b-hf \
      --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/llava-1.5-7b-hf.log 2>&1 &

# ===== Env: internvl - InternVL2-8B =====
(
  OUTFILE=$SAVE_DIR/InternVL2-8B.json
  LOGFILE=${LOG_DIR}/InternVL2-8B.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 50 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping InternVL2-8B - output exists and no errors: $OUTFILE"
  else
    conda activate work_dirs/envs/internvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python examiner/caption.py --dataset $DATASET --model_path OpenGVLab/InternVL2-8B \
      --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/InternVL2-8B.log 2>&1 &

# ===== Env: internvl - InternVL2.5-8B =====
(
  OUTFILE=$SAVE_DIR/InternVL2_5-8B.json
  LOGFILE=${LOG_DIR}/InternVL2_5-8B.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 50 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping InternVL2_5-8B - output exists and no errors: $OUTFILE"
  else
    conda activate work_dirs/envs/internvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python examiner/caption.py --dataset $DATASET --model_path OpenGVLab/InternVL2_5-8B \
      --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/InternVL2_5-8B.log 2>&1 &

# ===== Env: internvl - InternVL3-8B =====
(
  OUTFILE=$SAVE_DIR/InternVL3-8B-Instruct.json
  LOGFILE=${LOG_DIR}/InternVL3-8B-Instruct.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 50 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping InternVL3-8B-Instruct - output exists and no errors: $OUTFILE"
  else
    conda activate work_dirs/envs/internvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python examiner/caption.py --dataset $DATASET --model_path OpenGVLab/InternVL3-8B-Instruct \
      --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/InternVL3-8B-Instruct.log 2>&1 &

# ===== Env: qwenvl - Qwen2.5-VL-7B =====
(
  OUTFILE=$SAVE_DIR/Qwen2.5-VL-7B-Instruct.json
  LOGFILE=${LOG_DIR}/Qwen2.5-VL-7B-Instruct.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 50 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping Qwen2.5-VL-7B-Instruct - output exists and no errors: $OUTFILE"
  else
    conda activate work_dirs/envs/qwenvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python examiner/caption.py --dataset $DATASET --model_path Qwen/Qwen2.5-VL-7B-Instruct \
      --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/Qwen2.5-VL-7B-Instruct.log 2>&1 &

# ===== Env: gemma3 - gemma-3-12b-it =====
(
  OUTFILE=$SAVE_DIR/gemma-3-12b-it.json
  LOGFILE=${LOG_DIR}/gemma-3-12b-it.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 50 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping gemma-3-12b-it - output exists and no errors: $OUTFILE"
  else
    conda activate work_dirs/envs/gemma3
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python examiner/caption.py --dataset $DATASET --model_path google/gemma-3-12b-it \
      --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/gemma-3-12b-it.log 2>&1 &

# ===== Env: opera - opera-llava-1.5 =====
(
  OUTFILE=$SAVE_DIR/opera-llava-1.5.json
  LOGFILE=${LOG_DIR}/opera-llava-1.5.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 50 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping opera-llava-1.5 - output exists and no errors: $OUTFILE"
  else
    conda activate opera
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python examiner/caption.py --dataset $DATASET \
      --model_path /raid/william/project/context-eval-mllm/data/checkpoints/opera/llava-1.5 \
      --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/opera-llava-1.5.log 2>&1 &

wait

echo "All caption SVG jobs completed!"
echo "Outputs in : ${SAVE_DIR}"
echo "Logs in    : ${LOG_DIR}"
