#!/usr/bin/env bash
# Launch v19d91 (one-line CONTEXT_PROMPT tweak vs v19d9: cap selection at 10).
# 3 models x (N=1, 2, 5) = 9 jobs across 8 GPUs (8 in parallel + 1 queued).
# Outputs to work_dirs/svg500/v19d91-{variant}_gpt54/ (v19d9 outputs stay
# at work_dirs/svg500/v19d9-*_gpt54/ as the archive for comparison).

set -u
cd "$(dirname "$0")/../.."

export PYTHONPATH="./:infer:grader/easydetect:$(pwd)/infer/LLaVA:$(pwd)/infer/LLaVA/llava"

LAUNCH_DELAY=8

variant_dir() {
  case "$1" in
    1) echo "v19d91-1ctx-ablation_gpt54" ;;
    2) echo "v19d91-original_gpt54" ;;
    5) echo "v19d91-5ctx-ablation_gpt54" ;;
  esac
}

launch_job() {
  local gpu="$1" model_path="$2" model_short="$3" n="$4"
  local vdir; vdir="$(variant_dir "$n")"
  local outdir="work_dirs/svg500/${vdir}"
  local logdir="work_dirs/logs_svg500_${vdir}"
  mkdir -p "$outdir" "$logdir"
  local out="${outdir}/${model_short}.json"
  local cache="${outdir}/${model_short}_cache.json"
  local log="${logdir}/${model_short}.log"

  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u examiner/dyna_conv_v19d91.py \
    --dataset svg --num_samples 500 \
    --model_path "$model_path" \
    --num_contexts "$n" \
    --outfile "$out" \
    --cache_file "$cache" \
    --max_rounds 20 \
    > "$log" 2>&1 &
  echo "launched gpu=$gpu pid=$! n=$n model=$model_short -> $log"
  sleep "$LAUNCH_DELAY"
}

launch_job 0 "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL-7B-Instruct" 5
launch_job 3 "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL-7B-Instruct" 2
launch_job 6 "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL-7B-Instruct" 1

launch_job 1 "Qwen/Qwen3-VL-8B-Instruct" "Qwen3-VL-8B-Instruct" 5
launch_job 4 "Qwen/Qwen3-VL-8B-Instruct" "Qwen3-VL-8B-Instruct" 2
launch_job 7 "Qwen/Qwen3-VL-8B-Instruct" "Qwen3-VL-8B-Instruct" 1

launch_job 2 "llava-hf/llava-1.5-7b-hf" "llava-1.5-7b-hf" 5
launch_job 5 "llava-hf/llava-1.5-7b-hf" "llava-1.5-7b-hf" 2

# 9th (llava N=1) queued by v19d91_round2_queue_llava_n1.sh.
echo
echo "All 8 launched. 9th (llava-1.5 N=1) will be queued separately."
