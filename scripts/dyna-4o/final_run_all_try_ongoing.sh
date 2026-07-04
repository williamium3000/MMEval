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
# ===== Env: gemma3 - Gemma3 models =====

(
  conda activate work_dirs/envs/gemma3
  python $RUN_FILE --dataset vg --model_path google/gemma-3-4b-it --outfile $SAVE_DIR/gemma-3-4b-it.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/gemma-3-4b-it.log 2>&1 &


(
  conda activate work_dirs/envs/gemma3
  python $RUN_FILE --dataset vg --model_path google/gemma-3-12b-it --outfile $SAVE_DIR/gemma-3-12b-it.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/gemma-3-12b-it.log 2>&1 &

(
  conda activate work_dirs/envs/gemma3
  python $RUN_FILE --dataset vg --model_path google/gemma-3-27b-it --outfile $SAVE_DIR/gemma-3-27b-it.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/gemma-3-27b-it.log 2>&1 &



# Wait for all background jobs to complete
wait

echo "All parallel inference jobs completed!"
echo "Check logs in: ${LOG_DIR}"