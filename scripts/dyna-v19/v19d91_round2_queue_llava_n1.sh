#!/usr/bin/env bash
# Queue llava-1.5 N=1 for v19d91 once a GPU frees up.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="./:infer:grader/easydetect:$(pwd)/infer/LLaVA:$(pwd)/infer/LLaVA/llava"

OUTDIR="work_dirs/svg500/v19d91-1ctx-ablation_gpt54"
LOGDIR="work_dirs/logs_svg500_v19d91-1ctx-ablation_gpt54"
mkdir -p "$OUTDIR" "$LOGDIR"

OUT="${OUTDIR}/llava-1.5-7b-hf.json"
CACHE="${OUTDIR}/llava-1.5-7b-hf_cache.json"
LOG="${LOGDIR}/llava-1.5-7b-hf.log"

free_gpu() {
  for g in 6 7 3 4 5 0 1 2; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null)
    if [ -n "$free" ] && [ "$free" -ge 30000 ]; then
      echo "$g"; return
    fi
  done
}

# wait for the first 8 jobs to finish loading (so we don't double-allocate
# during the cold start when all GPUs still show free)
sleep 600

echo "polling for free GPU (>=30GB)..."
while true; do
  gpu=$(free_gpu)
  if [ -n "$gpu" ]; then
    echo "GPU $gpu free at $(date), launching llava-1.5 N=1 (v19d91)"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python -u examiner/dyna_conv_v19d91.py \
      --dataset svg --num_samples 500 \
      --model_path llava-hf/llava-1.5-7b-hf \
      --num_contexts 1 \
      --outfile "$OUT" \
      --cache_file "$CACHE" \
      --max_rounds 20 \
      > "$LOG" 2>&1 &
    echo "launched llava-1.5 N=1 (v19d91) pid=$! gpu=$gpu -> $LOG"
    break
  fi
  sleep 300
done
