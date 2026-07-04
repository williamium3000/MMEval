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

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0

# python examiners/dyna_conv_icl.py --model_path llava-hf/llava-1.5-7b-hf --dataset vg --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_COCO_ICL_UNANSWERABLE --outfile output/vg/dyna/coco/icl/icl_uq.json --icls icls/coco_icl_uq.json --num_samples 100
# python examiners/dyna_conv_icl.py --model_path llava-hf/llava-1.5-7b-hf --dataset vg --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_COCO_ICL_CERTAINTY_COVERAGE --outfile output/vg/dyna/coco/icl/icl_certainty_coverage.json --icls icls/coco_icl.json --num_samples 100 &
# python examiners/dyna_conv_icl.py --model_path llava-hf/llava-1.5-7b-hf --dataset vg --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_COCO_ICL_CERTAINTY --outfile output/vg/dyna/coco/icl/icl_certainty.json --icls icls/coco_icl.json --num_samples 100  &
# python examiners/dyna_conv_icl.py --model_path llava-hf/llava-1.5-7b-hf --dataset vg --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL_COVERAGE --outfile output/vg/dyna/coco/icl/icl_coverage.json --icls icls/icl.json --num_samples 100 &
# python examiners/dyna_conv_icl.py --model_path llava-hf/llava-1.5-7b-hf --dataset vg --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL --outfile output/vg/dyna/icl/icl.json --icls icls/icl.json --num_samples 100 &
# python examiners/dyna_conv_icl_context.py --model_path llava-hf/llava-1.5-7b-hf --dataset vg --p_mode None --outfile output/vg/dyna/icl/icl_context.json --num_samples 100


python examiners/dyna_conv_icl_context_coco.py --model_path llava-hf/llava-1.5-7b-hf --dataset coco --p_mode None --outfile output/vg/dyna/coco/icl/icl_context.json --num_samples 100