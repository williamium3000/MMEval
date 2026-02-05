#!/bin/bash -l
#SBATCH --job-name=parallel_inference
#SBATCH --time=12:0:0
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output "slurm_logs/slurm-%j.out"

export PYTHONPATH=$PYTHONPATH:./:infer:grader/easydetect
export PYTHONPATH="/raid/william/project/context-eval-mllm/infer/LLaVA:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=4,5,6,7

NUM_SAMPLES=100
SAVE_DIR=work_dirs/vg/final_run_v18_gpt4o
RUN_FILE=examiner/dyna_conv_v18.py
LOG_DIR=work_dirs/logs_try

# Create log directory
mkdir -p ${LOG_DIR}

# Initialize conda
eval "$(conda shell.bash hook)"

echo "Starting parallel inference jobs..."
echo "Logs will be saved to: ${LOG_DIR}"



# ===== Env: llava_next - LLaVA Next models =====
(
  conda activate /raid/ztw/envs/llava_next
  python $RUN_FILE --dataset vg --model_path llava-hf/llava-v1.6-vicuna-7b-hf --outfile $SAVE_DIR/llava-v1.6-vicuna-7b-hf.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/llava-v1.6-vicuna-7b-hf.log 2>&1 &

(
  conda activate /raid/ztw/envs/llava_next
  python $RUN_FILE --dataset vg --model_path llava-hf/llava-v1.6-vicuna-13b-hf --outfile $SAVE_DIR/llava-v1.6-vicuna-13b-hf.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/llava-v1.6-vicuna-13b-hf.log 2>&1 &

(
  conda activate /raid/ztw/envs/llava_next
  python $RUN_FILE --dataset vg --model_path llava-hf/llava-v1.6-34b-hf --outfile $SAVE_DIR/llava-v1.6-34b-hf.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/llava-v1.6-34b-hf.log 2>&1 &




(
  conda activate /raid/ztw/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL3-2B-Instruct --outfile $SAVE_DIR/InternVL3-2B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL3-2B-Instruct.log 2>&1 &

(
  conda activate /raid/ztw/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL3-8B-Instruct --outfile $SAVE_DIR/InternVL3-8B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL3-8B-Instruct.log 2>&1 &

(
  conda activate /raid/ztw/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL3-38B-Instruct --outfile $SAVE_DIR/InternVL3-38B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL3-38B-Instruct.log 2>&1 &


# ===== Env: phi3v - Phi3.5 models =====
(
  conda activate /raid/ztw/envs/phi3v
  python $RUN_FILE --dataset vg --model_path microsoft/Phi-3.5-vision-instruct --outfile $SAVE_DIR/Phi-3.5-vision-instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Phi-3.5-vision-instruct.log 2>&1 &


# ===== Env: phi4v - Phi4 models =====


# Wait for all background jobs to complete
wait

echo "All parallel inference jobs completed!"
echo "Check logs in: ${LOG_DIR}"