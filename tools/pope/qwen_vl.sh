#!/bin/bash -l
#SBATCH --job-name=test
#SBATCH --time=4:0:0
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output "slurm_logs/slurm-%j.out"


mkdir -p slurm_logs
OUTFILE=eval/pope/results/Qwen-VL-Chat/coco_pope_adversarial_result.json
python infer/infer_qwen_vl.py \
    --infile eval/pope/coco_pope_adversarial_converted.json \
    --outfile $OUTFILE \
    --img_dir data/coco/ \
    --model_path Qwen/Qwen-VL-Chat \

python eval/pope/eval.py $OUTFILE $OUTFILE 