#!/usr/bin/env bash
# Launch v19d91 N=1 variants for the 4 models currently running N=2.
# N=5 variants are launched by a separate queue script as GPUs free up
# (the N=2 + N=1 jobs eventually free their GPUs).
#
# Layout BEFORE: GPUs 3,4,5,6 have N=2 jobs alive; GPUs 0,1,2,7 idle.
# Layout AFTER:  N=1 jobs land on 0,1,2,7 immediately; N=5 jobs are
# queued and start as any GPU becomes free.

set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="./:infer:grader/easydetect:$(pwd)/infer/LLaVA:$(pwd)/infer/LLaVA/llava"

LAUNCH_DELAY=8
LOGDIR_N1="work_dirs/logs_svg500_v19d91-1ctx-ablation_gpt54"
OUTDIR_N1="work_dirs/svg500/v19d91-1ctx-ablation_gpt54"
mkdir -p "$OUTDIR_N1" "$LOGDIR_N1"

launch_n1() {
  local gpu="$1" model_path="$2" model_short="$3"
  local out="${OUTDIR_N1}/${model_short}.json"
  local cache="${OUTDIR_N1}/${model_short}_cache.json"
  local log="${LOGDIR_N1}/${model_short}.log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u examiner/dyna_conv_v19d91.py \
    --dataset svg --num_samples 500 \
    --model_path "$model_path" \
    --num_contexts 1 \
    --outfile "$out" \
    --cache_file "$cache" \
    --max_rounds 20 \
    > "$log" 2>&1 &
  echo "launched gpu=$gpu pid=$! N=1 model=$model_short -> $log"
  sleep "$LAUNCH_DELAY"
}

launch_n1 0 "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL-7B-Instruct"
launch_n1 1 "Qwen/Qwen3-VL-8B-Instruct"   "Qwen3-VL-8B-Instruct"
launch_n1 2 "llava-hf/llava-1.5-7b-hf"    "llava-1.5-7b-hf"
launch_n1 7 "google/gemma-3-12b-it"       "gemma-3-12b-it"

echo
echo "All 4 N=1 launched. Use v19d91_variants_queue_n5.sh to dispatch N=5 jobs as GPUs free."
