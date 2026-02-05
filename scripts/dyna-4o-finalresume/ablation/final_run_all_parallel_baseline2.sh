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
SAVE_DIR=work_dirs/vg/ablation_baseline2
RUN_FILE=examiner/dyna_conv.py
LOG_DIR=work_dirs/logs_ablation_baseline2

# Create log directory
mkdir -p ${LOG_DIR}

# Initialize conda
eval "$(conda shell.bash hook)"

echo "Starting parallel inference jobs..."
echo "Logs will be saved to: ${LOG_DIR}"

(
  OUTFILE=$SAVE_DIR/gemma-3-12b-it.json
  LOGFILE=${LOG_DIR}/gemma-3-12b-it.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping gemma-3-12b-it - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/gemma3
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path google/gemma-3-12b-it --outfile $OUTFILE --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL  --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/gemma-3-12b-it.log 2>&1 &


# ===== Env: internvl - InternVL2 Series =====
(
  OUTFILE=$SAVE_DIR/InternVL2-2B.json
  LOGFILE=${LOG_DIR}/InternVL2-2B.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping InternVL2-2B - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/internvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2-2B --outfile $OUTFILE --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL --num_samples $NUM_SAMPLES
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
    python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2-8B --outfile $OUTFILE --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/InternVL2-8B.log 2>&1 &

# (
#   OUTFILE=$SAVE_DIR/InternVL2-26B.json
#   LOGFILE=${LOG_DIR}/InternVL2-26B.log
#   if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
#     echo "Skipping InternVL2-26B - output file already exists and no errors in log: $OUTFILE"
#   else
#     conda activate work_dirs/envs/internvl
#     export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
#     python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2-26B --outfile $OUTFILE --num_samples $NUM_SAMPLES
#   fi
# ) > ${LOG_DIR}/InternVL2-26B.log 2>&1 &

# ===== Env: internvl - InternVL2.5 Series =====
(
  OUTFILE=$SAVE_DIR/InternVL2_5-2B.json
  LOGFILE=${LOG_DIR}/InternVL2_5-2B.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping InternVL2_5-2B - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/internvl
    export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
    python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2_5-2B --outfile $OUTFILE --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL --num_samples $NUM_SAMPLES
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
    python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2_5-8B --outfile $OUTFILE --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/InternVL2_5-8B.log 2>&1 &

# Wait for all background jobs to complete
wait

echo "All parallel inference jobs completed!"
echo "Check logs in: ${LOG_DIR}"