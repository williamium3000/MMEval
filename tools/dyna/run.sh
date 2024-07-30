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
# python examiners/dyna_conv.py --debug --p_mode coverage --outfile output/dyna_bad_examples/coverage.json
# python visualize.py output/dyna_bad_examples/coverage_model_perspective.json visualization/dyna_bad_examples/coverage_model_perspective --anno 


python examiners/dyna_conv_json.py --debug --use_json_mode --p_mode coverage_certainty_with_answer --outfile output/dyna_bad_examples/coverage_certainty_with_answer_json_mode.json
python visualize.py output/dyna_bad_examples/coverage_certainty_with_answer_json_mode.json visualization/dyna_bad_examples/coverage_certainty_with_answer_json_mode --anno 