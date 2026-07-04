export PYTHONPATH=$PYTHONPATH:./:infer:grader/easydetect
export CUDA_VISIBLE_DEVICES=0

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate work_dirs/envs/qwenvl2d5

python examiner/dyna_conv_v6.py \
    --dataset vg --model_path llava-hf/llava-1.5-7b-hf  \
    --outfile work_dirs/vg/dyna-v6/llava-1.5-7b-hf.json \
    --num_samples 20

python examiner/dyna_conv_v6.py \
    --dataset vg --model_path llava-hf/llava-1.5-13b-hf  \
    --outfile work_dirs/vg/dyna-v6/llava-1.5-13b-hf.json \
    --num_samples 20