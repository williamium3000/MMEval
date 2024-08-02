export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
python graders/mmal/mmhal_grader.py \
    --response output/caption/caption_converted-mmal.json \
    --evaluation output/graders/mmhal/caption.json