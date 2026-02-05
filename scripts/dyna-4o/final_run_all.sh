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

export PYTHONPATH=$PYTHONPATH:./:infer:grader/easydetect
export CUDA_VISIBLE_DEVICES=3

NUM_SAMPLES=100
SAVE_DIR=work_dirs/vg/final_run_v18_gpt4o
RUN_FILE=examiner/dyna_conv_v18.py

# Initialize conda
eval "$(conda shell.bash hook)"

conda activate /raid/miniconda3/envs/qwenvl

# ===== Qwen2.5-VL Series =====
# Env: qwenvl (same as Qwen2-VL)
python $RUN_FILE \
    --dataset vg --model_path Qwen/Qwen2.5-VL-3B-Instruct  \
    --outfile $SAVE_DIR/Qwen2.5-VL-3B-Instruct.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path Qwen/Qwen2.5-VL-7B-Instruct  \
    --outfile $SAVE_DIR/Qwen2.5-VL-7B-Instruct.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path Qwen/Qwen2.5-VL-72B-Instruct  \
    --outfile $SAVE_DIR/Qwen2.5-VL-72B-Instruct.json \
    --num_samples $NUM_SAMPLES

# ===== Qwen3-VL Series =====
# Env: qwenvl (NOTE: not in registry, using qwenvl)
python $RUN_FILE \
    --dataset vg --model_path Qwen/Qwen3-VL-2B-Instruct  \
    --outfile $SAVE_DIR/Qwen3-VL-2B-Instruct.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path Qwen/Qwen3-VL-8B-Instruct  \
    --outfile $SAVE_DIR/Qwen3-VL-8B-Instruct.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path Qwen/Qwen3-VL-32B-Instruct  \
    --outfile $SAVE_DIR/Qwen3-VL-32B-Instruct.json \
    --num_samples $NUM_SAMPLES

# ===== LLaVA Series =====
# Env: llava
conda activate work_dirs/envs/llava

python $RUN_FILE \
    --dataset vg --model_path llava-hf/llava-1.5-7b-hf  \
    --outfile $SAVE_DIR/llava-1.5-7b-hf.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path llava-hf/llava-1.5-13b-hf  \
    --outfile $SAVE_DIR/llava-1.5-13b-hf.json \
    --num_samples $NUM_SAMPLES

# ===== LLaVA Next Series =====
# Env: llava_next
conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/llava_next

python $RUN_FILE \
    --dataset vg --model_path llava-hf/llava-v1.6-vicuna-7b-hf  \
    --outfile $SAVE_DIR/llava-v1.6-vicuna-7b-hf.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path llava-hf/llava-v1.6-vicuna-13b-hf  \
    --outfile $SAVE_DIR/llava-v1.6-vicuna-13b-hf.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path llava-hf/llava-v1.6-34b-hf  \
    --outfile $SAVE_DIR/llava-v1.6-34b-hf.json \
    --num_samples $NUM_SAMPLES

# ===== LLaVA OneVision Series =====
# Env: llava_ov
conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/llava_ov

python $RUN_FILE \
    --dataset vg --model_path llava-hf/llava-onevision-qwen2-0.5b-ov-hf  \
    --outfile $SAVE_DIR/llava-onevision-qwen2-0.5b-ov-hf.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path llava-hf/llava-onevision-qwen2-7b-ov-hf  \
    --outfile $SAVE_DIR/llava-onevision-qwen2-7b-ov-hf.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path llava-hf/llava-onevision-qwen2-72b-ov-hf  \
    --outfile $SAVE_DIR/llava-onevision-qwen2-72b-ov-hf.json \
    --num_samples $NUM_SAMPLES



# ===== InternVL2 Series =====
# Env: internvl
conda activate work_dirs/envs/internvl

python $RUN_FILE \
    --dataset vg --model_path OpenGVLab/InternVL2-2B  \
    --outfile $SAVE_DIR/InternVL2-2B.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path OpenGVLab/InternVL2-8B  \
    --outfile $SAVE_DIR/InternVL2-8B.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path OpenGVLab/InternVL2-26B  \
    --outfile $SAVE_DIR/InternVL2-26B.json \
    --num_samples $NUM_SAMPLES

# ===== InternVL2.5 Series =====
# Env: internvl (same as InternVL2)
python $RUN_FILE \
    --dataset vg --model_path OpenGVLab/InternVL2_5-2B  \
    --outfile $SAVE_DIR/InternVL2_5-2B.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path OpenGVLab/InternVL2_5-8B  \
    --outfile $SAVE_DIR/InternVL2_5-8B.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path OpenGVLab/InternVL2_5-38B  \
    --outfile $SAVE_DIR/InternVL2_5-38B.json \
    --num_samples $NUM_SAMPLES

# ===== InternVL3 Series =====
# Env: internvl (same as InternVL2)
python $RUN_FILE \
    --dataset vg --model_path OpenGVLab/InternVL3-2B-Instruct  \
    --outfile $SAVE_DIR/InternVL3-2B-Instruct.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path OpenGVLab/InternVL3-8B-Instruct  \
    --outfile $SAVE_DIR/InternVL3-8B-Instruct.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path OpenGVLab/InternVL3-38B-Instruct  \
    --outfile $SAVE_DIR/InternVL3-38B-Instruct.json \
    --num_samples $NUM_SAMPLES

# ===== BLIP2 Series =====
# Env: blip2_flan_t5
conda activate work_dirs/envs/flan_t5

python $RUN_FILE \
    --dataset vg --model_path Salesforce/blip2-flan-t5-xl  \
    --outfile $SAVE_DIR/blip2-flan-t5-xl.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path Salesforce/blip2-flan-t5-xxl  \
    --outfile $SAVE_DIR/blip2-flan-t5-xxl.json \
    --num_samples $NUM_SAMPLES

# ===== InstructBLIP Series =====
# Env: instructblip
conda activate work_dirs/envs/instructblip

python $RUN_FILE \
    --dataset vg --model_path Salesforce/instructblip-vicuna-7b  \
    --outfile $SAVE_DIR/instructblip-vicuna-7b.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path Salesforce/instructblip-vicuna-13b  \
    --outfile $SAVE_DIR/instructblip-vicuna-13b.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path Salesforce/instructblip-flan-t5-xxl  \
    --outfile $SAVE_DIR/instructblip-flan-t5-xxl.json \
    --num_samples $NUM_SAMPLES

# ===== Gemma3 Series =====
# Env: gemma3
conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/gemma3

python $RUN_FILE \
    --dataset vg --model_path google/gemma-3-4b-it  \
    --outfile $SAVE_DIR/gemma-3-4b-it.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path google/gemma-3-12b-it  \
    --outfile $SAVE_DIR/gemma-3-12b-it.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path google/gemma-3-27b-it  \
    --outfile $SAVE_DIR/gemma-3-27b-it.json \
    --num_samples $NUM_SAMPLES

# ===== Cambrian Series =====
# Env: cambrian
conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/cambrian

python $RUN_FILE \
    --dataset vg --model_path nyu-visionx/cambrian-phi3-3b  \
    --outfile $SAVE_DIR/cambrian-phi3-3b.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path nyu-visionx/cambrian-8b  \
    --outfile $SAVE_DIR/cambrian-8b.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path nyu-visionx/cambrian-34b  \
    --outfile $SAVE_DIR/cambrian-34b.json \
    --num_samples $NUM_SAMPLES

# ===== Janus Pro Series =====
# Env: janus
conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/janus

python $RUN_FILE \
    --dataset vg --model_path deepseek-ai/Janus-Pro-1B  \
    --outfile $SAVE_DIR/Janus-Pro-1B.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path deepseek-ai/Janus-Pro-7B  \
    --outfile $SAVE_DIR/Janus-Pro-7B.json \
    --num_samples $NUM_SAMPLES

# ===== Mantis Series =====
# Env: mantis
conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/mantis

python $RUN_FILE \
    --dataset vg --model_path TIGER-Lab/Mantis-8B-clip-llama3  \
    --outfile $SAVE_DIR/Mantis-8B-clip-llama3.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path TIGER-Lab/Mantis-8B-siglip-llama3  \
    --outfile $SAVE_DIR/Mantis-8B-siglip-llama3.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path TIGER-Lab/Mantis-8B-Idefics2  \
    --outfile $SAVE_DIR/Mantis-8B-Idefics2.json \
    --num_samples $NUM_SAMPLES

# ===== Ovis1.5 Series =====
# Env: ovis1d5
conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/ovis1d5

python $RUN_FILE \
    --dataset vg --model_path AIDC-AI/Ovis1.5-Llama3-8B  \
    --outfile $SAVE_DIR/Ovis1.5-Llama3-8B.json \
    --num_samples $NUM_SAMPLES

python $RUN_FILE \
    --dataset vg --model_path AIDC-AI/Ovis1.5-Gemma2-9B  \
    --outfile $SAVE_DIR/Ovis1.5-Gemma2-9B.json \
    --num_samples $NUM_SAMPLES

# ===== Phi3.5 Series =====
# Env: phi3v
conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/phi3v

python $RUN_FILE \
    --dataset vg --model_path microsoft/Phi-3.5-vision-instruct  \
    --outfile $SAVE_DIR/Phi-3.5-vision-instruct.json \
    --num_samples $NUM_SAMPLES