export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
unset all_proxy

python graders/mmhal/mmhal_grader.py \
    --response output/dyna_bad_examples/coverage_certainty_with_answer_start_with_desc_json_mode.json \
    --evaluation output/graders/mmhal/coverage_certainty_with_answer_start_with_desc_json_mode.json