#!/bin/bash -l
#SBATCH --job-name=test
#SBATCH --time=4:0:0
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output "slurm_logs/slurm-%j.out"

# mkdir -p slurm_logs
# conda activate llava
export PYTHONPATH=$PYTHONPATH:./:infer:grader/easydetect
export CUDA_VISIBLE_DEVICES=7

NUM_SAMPLES=20
SAVE_DIR=work_dirs/vg/dyna_conv_v5_v3
RUN_FILE=examiner/dyna_conv_v5_v3.py
# Initialize conda
eval "$(conda shell.bash hook)"
conda activate /raid/miniconda3/envs/qwenvl

# python $RUN_FILE \
#     --dataset vg --model_path llava-hf/llava-1.5-7b-hf  \
#     --outfile $SAVE_DIR/llava-1.5-7b-hf.json \
#     --num_samples $NUM_SAMPLES

# python $RUN_FILE \
#     --dataset vg --model_path llava-hf/llava-1.5-13b-hf  \
#     --outfile $SAVE_DIR/llava-1.5-13b-hf.json \
#     --num_samples $NUM_SAMPLES


python $RUN_FILE \
    --dataset vg --model_path Qwen/Qwen2.5-VL-3B-Instruct  \
    --outfile $SAVE_DIR/Qwen2.5-VL-3B-Instruct.json \
    --num_samples $NUM_SAMPLES

# python $RUN_FILE \
#     --dataset vg --model_path Salesforce/blip2-flan-t5-xl  \
#     --outfile $SAVE_DIR/blip2-flan-t5-xl.json \
#     --num_samples $NUM_SAMPLES

# python $RUN_FILE \
#     --dataset vg --model_path Salesforce/blip2-flan-t5-xxl  \
#     --outfile $SAVE_DIR/blip2-flan-t5-xxl.json \
#     --num_samples $NUM_SAMPLES


# python $RUN_FILE \
#     --dataset vg --model_path Salesforce/instructblip-vicuna-7b \
#     --outfile $SAVE_DIR/instructblip-vicuna-7b.json \
#     --num_samples $NUM_SAMPLES


# conda activate opera

# python $RUN_FILE \
#     --dataset vg --model_path opera/llava-1.5 \
#     --outfile $SAVE_DIR/opera-llava-1.5.json \
#     --num_samples $NUM_SAMPLES


# conda activate work_dirs/envs/phi4

# python $RUN_FILE \
#     --dataset vg --model_path OpenGVLab/InternVL3-8B-Instruct \
#     --outfile $SAVE_DIR/InternVL3-8B-Instruct.json \
#     --num_samples $NUM_SAMPLES

# python $RUN_FILE \
#     --dataset vg --model_path OpenGVLab/InternVL2_5-8B \
#     --outfile $SAVE_DIR/InternVL2_5-8B.json \
#     --num_samples $NUM_SAMPLES

# python $RUN_FILE \
#     --dataset vg --model_path OpenGVLab/InternVL2-8B \
#     --outfile $SAVE_DIR/InternVL2-8B.json \
#     --num_samples $NUM_SAMPLES


# python $RUN_FILE \
#     --dataset vg --model_path microsoft/Phi-3.5-vision-instruct \
#     --outfile $SAVE_DIR/Phi-3.5-vision-instruct.json \
#     --num_samples $NUM_SAMPLES


# python $RUN_FILE \
#     --dataset vg --model_path data/checkpoints/idefics2-8b-lpoi-list5-10k/final \
#     --outfile $SAVE_DIR/idefics2-8b-lpoi-list5-10k.json \
#     --num_samples $NUM_SAMPLES

# conda activate work_dirs/envs/gemma3

# python $RUN_FILE \
#     --dataset vg --model_path google/gemma-3-4b-it\
#     --outfile $SAVE_DIR/gemma-3-4b-it.json \
#     --num_samples $NUM_SAMPLES

# python $RUN_FILE \
#     --dataset vg --model_path google/paligemma-3b-mix-224\
#     --outfile $SAVE_DIR/paligemma-3b-mix-224.json \
#     --num_samples $NUM_SAMPLES

# eval "$(conda shell.bash hook)"
# conda activate work_dirs/envs/llava

# python $RUN_FILE \
#     --dataset vg --model_path data/checkpoints/LLaVA-RLHF-13b-v1.5-336\
#     --outfile $SAVE_DIR/LLaVA-RLHF-13b-v1.5-336.json \
#     --num_samples $NUM_SAMPLES

# conda activate work_dirs/envs/ovis2
# python $RUN_FILE \
#     --dataset vg --model_path AIDC-AI/Ovis2-8B \
#     --outfile $SAVE_DIR/Ovis2-8B.json \
#     --num_samples $NUM_SAMPLES



