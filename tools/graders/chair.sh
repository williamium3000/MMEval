export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0

python graders/chair/convert.py output/dyna_bad_examples/coverage_certainty_with_answer_json_mode.json
python graders/chair/chair.py graders/chair/output/coverage_certainty_with_answer_json_mode.json