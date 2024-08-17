export PYTHONPATH=./
python examiners/caption.py --debug --outfile output/caption_200/caption.json
python examiners/dyna_conv_json.py --debug --p_mode coverage_certainty_with_answer --outfile output/dyna_bad_examples_200/coverage_certainty_with_answer_json_mode.json
python examiners/dyna_conv.py --debug --p_mode coverage_certainty --outfile output/dyna_bad_examples_200/coverage_certainty.json

# split the CKPT and obtain the last directory as CKPT_NAME
CKPT=work_dirs/objectllama/sft/finetune_objectllama_full-ft_vllava-sft
CKPT_NAME=$(echo $CKPT | tr "/" "\n" | tail -1)
# MODEL_BASE default to None
MODEL_BASE=${2: -None}
