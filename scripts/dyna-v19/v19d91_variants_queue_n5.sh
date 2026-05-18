#!/usr/bin/env bash
# Queue 4 v19d91 N=5 jobs (Qwen2.5-VL, Qwen3-VL, llava-1.5, gemma-3-12b)
# to launch as GPUs free up. Each picks the lowest-numbered free GPU
# (>=30GB free) and starts when one is available.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="./:infer:grader/easydetect:$(pwd)/infer/LLaVA:$(pwd)/infer/LLaVA/llava"

OUTDIR="work_dirs/svg500/v19d91-5ctx-ablation_gpt54"
LOGDIR="work_dirs/logs_svg500_v19d91-5ctx-ablation_gpt54"
mkdir -p "$OUTDIR" "$LOGDIR"

# Models still to launch (model_path|model_short). The script pops one
# at a time as GPUs become free.
QUEUE=(
  "Qwen/Qwen2.5-VL-7B-Instruct|Qwen2.5-VL-7B-Instruct"
  "Qwen/Qwen3-VL-8B-Instruct|Qwen3-VL-8B-Instruct"
  "llava-hf/llava-1.5-7b-hf|llava-1.5-7b-hf"
  "google/gemma-3-12b-it|gemma-3-12b-it"
)

free_gpu() {
  # Strict: GPU is "free" only if memory.used < 1000 MiB (no model loaded).
  # The earlier 30GB-free heuristic stacked N=5 jobs onto GPUs that still
  # had N=1 models loaded, crushing throughput. Wait for actual idle.
  for g in 0 1 2 7 3 4 5 6; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
    if [ -n "$used" ] && [ "$used" -lt 1000 ]; then
      echo "$g"; return
    fi
  done
}

# Skip the 10-min sleep — at re-launch time all 8 existing jobs already
# have models loaded, so no risk of double-allocating during cold start.

for item in "${QUEUE[@]}"; do
  model_path="${item%%|*}"
  model_short="${item##*|}"

  echo "[$(date)] waiting for free GPU to launch N=5 $model_short..."
  while true; do
    gpu=$(free_gpu)
    if [ -n "$gpu" ]; then
      out="${OUTDIR}/${model_short}.json"
      cache="${OUTDIR}/${model_short}_cache.json"
      log="${LOGDIR}/${model_short}.log"
      CUDA_VISIBLE_DEVICES="$gpu" nohup python -u examiner/dyna_conv_v19d91.py \
        --dataset svg --num_samples 500 \
        --model_path "$model_path" \
        --num_contexts 5 \
        --outfile "$out" \
        --cache_file "$cache" \
        --max_rounds 20 \
        > "$log" 2>&1 &
      echo "[$(date)] launched N=5 $model_short pid=$! gpu=$gpu -> $log"
      # Give the new job ~5 min to allocate its GPU before the next poll
      sleep 300
      break
    fi
    sleep 300
  done
done

echo "[$(date)] all 4 N=5 jobs launched."
