eval "$(conda shell.bash hook)"
conda activate work_dirs/envs/qwenvl3
CUDA_VISIBLE_DEVICES=5 python infer/infer_qwenvl3.py \
    --infile work_dirs/pope/questions/Qwen3-VL-8B-Instruct_pope_dsg_qa_extracted.json \
    --outfile work_dirs/pope/results/Qwen3-VL-8B-Instruct_pope_dsg_qa_extracted.json \
    --img_dir "" \
    --model_path Qwen/Qwen3-VL-8B-Instruct

# work_dirs/pope/questions/InternVL3-8B-Instruct_pope.json
conda activate work_dirs/envs/internvl
python infer/infer_internvl3.py \
    --infile work_dirs/pope/questions/InternVL3-8B-Instruct_pope_dsg_qa_extracted.json \
    --outfile work_dirs/pope/results/InternVL3-8B-Instruct_pope_dsg_qa_extracted.json \
    --img_dir data/coco/val2017 \
    --model_path OpenGVLab/InternVL3-8B-Instruct

# work_dirs/pope/questions/llava-1.5-7b-hf_single_pope.json
conda activate work_dirs/envs/llava
python infer/infer_llava.py \
    --infile work_dirs/pope/questions/llava-1.5-7b-hf_single_pope_dsg_qa_extracted.json \
    --outfile work_dirs/pope/results/llava-1.5-7b-hf_single_pope_dsg_qa_extracted.json \
    --img_dir data/coco/val2017 \
    --model_path liuhaotian/llava-v1.5-7b 