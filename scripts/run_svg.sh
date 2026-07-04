export PYTHONPATH=./
python examiner/caption.py --dataset svg --num_samples 100 --outfile output/caption_200/caption_svg.json
python examiner/caption.py --dataset svg --num_samples 5 --outfile output/caption_200/caption_svg.json

 python examiner/dyna_conv_v6.py \
  --dataset svg \
  --num_samples 5 \
  --model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
  --outfile output/dyna_conv_v6_svg.json