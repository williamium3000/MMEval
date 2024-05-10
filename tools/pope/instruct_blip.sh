#!/bin/bash -l
#SBATCH --job-name=test
#SBATCH --time=48:0:0
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output "slurm_logs/slurm-%j.out"

conda activate llava
mkdir -p slurm_logs

python infer/llava.py \
    --infile eval/pope/coco_pope_adversarial_converted.json \
    --outfile eval/pope/results/llava-v1.5-7b/coco_pope_adversarial_result.json \
    --img_dir data/coco/ \
    --model_path liuhaotian/llava-v1.5-7b \
