#!/usr/bin/env bash
# Launch round-2 (Qwen2.5-VL / Qwen3-VL / llava-1.5) × (N=1, 2, 5) SVG-500 sweep.
# 8 jobs in parallel on GPUs 0-7; 9th (llava-1.5 N=1) queued in a separate
# script after a GPU frees up.
#
# Assumes:
#   - cwd = repo root
#   - .env has AZURE_* creds + HF_TOKEN (utils/llm.py loads it via dotenv)
#   - The examiner script reads --num_contexts and routes via azure/ prefix
#     for the api examiner (we don't use it here; examinees are local).

set -u
cd "$(dirname "$0")/../.."

export PYTHONPATH="./:infer:grader/easydetect:$(pwd)/infer/LLaVA:$(pwd)/infer/LLaVA/llava"

LAUNCH_DELAY=8   # stagger so HF cache locking doesn't pile up

variant_dir() {
  case "$1" in
    1) echo "v19d9-1ctx-ablation_gpt54" ;;
    2) echo "v19d9-original_gpt54" ;;
    5) echo "v19d9-5ctx-ablation_gpt54" ;;
    *) echo "v19d9-${1}ctx-ablation_gpt54" ;;
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

  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u examiner/dyna_conv_v19d9.py \
    --dataset svg --num_samples 500 \
    --model_path "$model_path" \
    --num_contexts "$n" \
    --outfile "$out" \
    --cache_file "$cache" \
    --max_rounds 20 \
    > "$log" 2>&1 &
  local pid=$!
  echo "launched gpu=$gpu pid=$pid n=$n model=$model_short -> $log"
  sleep "$LAUNCH_DELAY"
}

# GPU 0/3/6: Qwen2.5-VL (already cached)
launch_job 0 "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL-7B-Instruct" 5
launch_job 3 "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL-7B-Instruct" 2
launch_job 6 "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL-7B-Instruct" 1

# GPU 1/4/7: Qwen3-VL (will download on first launch ~20GB)
launch_job 1 "Qwen/Qwen3-VL-8B-Instruct" "Qwen3-VL-8B-Instruct" 5
launch_job 4 "Qwen/Qwen3-VL-8B-Instruct" "Qwen3-VL-8B-Instruct" 2
launch_job 7 "Qwen/Qwen3-VL-8B-Instruct" "Qwen3-VL-8B-Instruct" 1

# GPU 2/5: llava-1.5 (will download ~15GB)
launch_job 2 "llava-hf/llava-1.5-7b-hf" "llava-1.5-7b-hf" 5
launch_job 5 "llava-hf/llava-1.5-7b-hf" "llava-1.5-7b-hf" 2

# 9th job (llava-1.5 N=1) is queued via a separate watcher — see v19d9_round2_queue_llava_n1.sh
echo
echo "All 8 launched. 9th (llava-1.5 N=1) will be queued separately."
echo "Tail logs:"
ls work_dirs/logs_svg500_*/*.log 2>/dev/null
