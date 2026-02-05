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

(
  OUTFILE=$SAVE_DIR/Qwen3-VL-32B-Instruct.json
  LOGFILE=${LOG_DIR}/Qwen3-VL-32B-Instruct.log
  if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
    echo "Skipping Qwen3-VL-32B-Instruct - output file already exists and no errors in log: $OUTFILE"
  else
    conda activate work_dirs/envs/qwenvl3
    export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
    python $RUN_FILE --dataset vg --model_path Qwen/Qwen3-VL-32B-Instruct --outfile $OUTFILE --num_samples $NUM_SAMPLES
  fi
) > ${LOG_DIR}/Qwen3-VL-32B-Instruct.log 2>&1 &

# (
#   OUTFILE=$SAVE_DIR/InternVL2_5-38B.json
#   LOGFILE=${LOG_DIR}/InternVL2_5-38B.log
#   if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
#     echo "Skipping InternVL2_5-38B - output file already exists and no errors in log: $OUTFILE"
#   else
#     conda activate work_dirs/envs/internvl
#     export CUDA_VISIBLE_DEVICES=$(select_gpus 1)
#     python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2_5-38B --outfile $OUTFILE --num_samples $NUM_SAMPLES
#   fi
# ) > ${LOG_DIR}/InternVL2_5-38B.log 2>&1 &
# (
#   conda activate work_dirs/envs/qwenvl
#   export CUDA_VISIBLE_DEVICES=$(select_gpus 2)
#   python $RUN_FILE --dataset vg --model_path Qwen/Qwen2.5-VL-72B-Instruct --outfile $SAVE_DIR/Qwen2.5-VL-72B-Instruct.json --num_samples $NUM_SAMPLES
# ) > ${LOG_DIR}/Qwen2.5-VL-72B-Instruct.log 2>&1 &

# Wait for all background jobs to complete
wait

echo "All parallel inference jobs completed!"
echo "Check logs in: ${LOG_DIR}"