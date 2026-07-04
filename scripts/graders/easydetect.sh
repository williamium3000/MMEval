export PYTHONPATH=$PYTHONPATH:./:infer:grader/easydetect

cd grader/easydetect
python test.py \
    --json /mnt/data/william/project/context-eval-mllm/work_dirs/vg/caption_test/llava-1.5-7b-hf.json \
    --outfile /mnt/data/william/project/context-eval-mllm/work_dirs/vg/caption_test/llava-1.5-7b-hf_easydetect.json