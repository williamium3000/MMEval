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
LOG_DIR=work_dirs/logs

# Create log directory
mkdir -p ${LOG_DIR}

# Initialize conda
eval "$(conda shell.bash hook)"

echo "Starting parallel inference jobs..."
echo "Logs will be saved to: ${LOG_DIR}"

# ===== Env: qwenvl - Qwen models =====
# ===== Env: llava - LLaVA models =====
(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/llava
  export PYTHONPATH="./infer/LLaVA:${PYTHONPATH:-}"
  python $RUN_FILE --dataset vg --model_path llava-hf/llava-1.5-7b-hf --outfile $SAVE_DIR/llava-1.5-7b-hf.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/llava-1.5-7b-hf.log 2>&1 &

(
  conda activate work_dirs/envs/llava
  export PYTHONPATH="./infer/LLaVA:${PYTHONPATH:-}"
  python $RUN_FILE --dataset vg --model_path llava-hf/llava-1.5-13b-hf --outfile $SAVE_DIR/llava-1.5-13b-hf.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/llava-1.5-13b-hf.log 2>&1 &

(
  conda activate envs/llava
  export PYTHONPATH="./infer/LLaVA:${PYTHONPATH:-}"
  python $RUN_FILE --dataset vg --model_path data/checkpoints/LLaVA-RLHF-13b-v1.5-336 --outfile $SAVE_DIR/LLaVA-RLHF-13b-v1.5-336.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/LLaVA-RLHF-13b-v1.5-336.log 2>&1 &


# ===== Env: internvl - InternVL models =====
(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2-2B --outfile $SAVE_DIR/InternVL2-2B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2-2B.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2-8B --outfile $SAVE_DIR/InternVL2-8B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2-8B.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2-26B --outfile $SAVE_DIR/InternVL2-26B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2-26B.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2_5-2B --outfile $SAVE_DIR/InternVL2_5-2B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2_5-2B.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2_5-8B --outfile $SAVE_DIR/InternVL2_5-8B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2_5-8B.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2_5-38B --outfile $SAVE_DIR/InternVL2_5-38B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2_5-38B.log 2>&1 &

# ===== Env: instructblip - InstructBLIP models =====
(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/instructblip
  python $RUN_FILE --dataset vg --model_path Salesforce/instructblip-vicuna-13b --outfile $SAVE_DIR/instructblip-vicuna-13b.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/instructblip-vicuna-13b.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/instructblip
  python $RUN_FILE --dataset vg --model_path Salesforce/instructblip-flan-t5-xxl --outfile $SAVE_DIR/instructblip-flan-t5-xxl.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/instructblip-flan-t5-xxl.log 2>&1 &

(
  conda activate /raid/miniconda3/envs/qwenvl
  python $RUN_FILE --dataset vg --model_path Qwen/Qwen2.5-VL-7B-Instruct --outfile $SAVE_DIR/Qwen2.5-VL-7B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Qwen2.5-VL-7B-Instruct.log 2>&1 &

(
  conda activate /raid/miniconda3/envs/qwenvl
  python $RUN_FILE --dataset vg --model_path Qwen/Qwen2.5-VL-72B-Instruct --outfile $SAVE_DIR/Qwen2.5-VL-72B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Qwen2.5-VL-72B-Instruct.log 2>&1 &


# ===== Env: phi4v - Phi4 models =====
# # ===== Env: qwenvl - Qwen models =====

# (
#   conda activate  /raid/ztw/envs/qwenvl3
#   python $RUN_FILE --dataset vg --model_path Qwen/Qwen3-VL-2B-Instruct --outfile $SAVE_DIR/Qwen3-VL-2B-Instruct.json --num_samples $NUM_SAMPLES
# ) > ${LOG_DIR}/Qwen3-VL-2B-Instruct.log 2>&1 &

# (
#   conda activate  /raid/ztw/envs/qwenvl3
#   python $RUN_FILE --dataset vg --model_path Qwen/Qwen3-VL-8B-Instruct --outfile $SAVE_DIR/Qwen3-VL-8B-Instruct.json --num_samples $NUM_SAMPLES
# ) > ${LOG_DIR}/Qwen3-VL-8B-Instruct.log 2>&1 &

# (
#   conda activate  /raid/ztw/envs/qwenvl3
#   python $RUN_FILE --dataset vg --model_path Qwen/Qwen3-VL-32B-Instruct --outfile $SAVE_DIR/Qwen3-VL-32B-Instruct.json --num_samples $NUM_SAMPLES
# ) > ${LOG_DIR}/Qwen3-VL-32B-Instruct.log 2>&1 &


# Wait for all background jobs to complete
wait

echo "All parallel inference jobs completed!"
echo "Check logs in: ${LOG_DIR}"