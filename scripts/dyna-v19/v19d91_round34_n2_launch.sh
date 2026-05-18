#!/usr/bin/env bash
# Launch v19d91 N=2 on InternVL2/2.5/3 + gemma-3 (rounds 3-4 of the original
# CEDI plan). Uses GPUs 0, 1, 2, 6 (GPU 7 stays idle); GPUs 3, 4, 5 are
# already occupied by the Qwen/llava v19d91 N=2 jobs.

set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="./:infer:grader/easydetect:$(pwd)/infer/LLaVA:$(pwd)/infer/LLaVA/llava"

LAUNCH_DELAY=8
OUTDIR="work_dirs/svg500/v19d91-original_gpt54"
LOGDIR="work_dirs/logs_svg500_v19d91-original_gpt54"
mkdir -p "$OUTDIR" "$LOGDIR"

launch_n2() {
  local gpu="$1" model_path="$2" model_short="$3"
  local out="${OUTDIR}/${model_short}.json"
  local cache="${OUTDIR}/${model_short}_cache.json"
  local log="${LOGDIR}/${model_short}.log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u examiner/dyna_conv_v19d91.py \
    --dataset svg --num_samples 500 \
    --model_path "$model_path" \
    --num_contexts 2 \
    --outfile "$out" \
    --cache_file "$cache" \
    --max_rounds 20 \
    > "$log" 2>&1 &
  echo "launched gpu=$gpu pid=$! model=$model_short -> $log"
  sleep "$LAUNCH_DELAY"
}

launch_n2 0 "OpenGVLab/InternVL2-8B"          "InternVL2-8B"
launch_n2 1 "OpenGVLab/InternVL2_5-8B"        "InternVL2_5-8B"
launch_n2 2 "OpenGVLab/InternVL3-8B-Instruct" "InternVL3-8B-Instruct"
launch_n2 6 "google/gemma-3-12b-it"           "gemma-3-12b-it"

echo
echo "Launched 4 N=2 jobs. GPU 7 stays idle."
