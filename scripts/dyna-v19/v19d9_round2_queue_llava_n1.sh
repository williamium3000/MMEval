#!/usr/bin/env bash
# Wait for one of the round-2 jobs to finish, then launch llava-1.5 N=1 on
# its freed GPU. Expected to fire ~18-20h after launch (one of the N=1 jobs
# on GPU 6 or 7 should finish first).
set -u
cd "$(dirname "$0")/../.."

export PYTHONPATH="./:infer:grader/easydetect:$(pwd)/infer/LLaVA:$(pwd)/infer/LLaVA/llava"

OUTDIR="work_dirs/svg500/v19d9-1ctx-ablation_gpt54"
LOGDIR="work_dirs/logs_svg500_v19d9-1ctx-ablation_gpt54"
mkdir -p "$OUTDIR" "$LOGDIR"

OUT="${OUTDIR}/llava-1.5-7b-hf.json"
CACHE="${OUTDIR}/llava-1.5-7b-hf_cache.json"
LOG="${LOGDIR}/llava-1.5-7b-hf.log"

free_gpu() {
  # Print the first GPU index with no examiner process running on it.
  for g in 6 7 3 4 5 0 1 2; do
    if ! pgrep -af "CUDA_VISIBLE_DEVICES=${g}.*dyna_conv_v19d9" > /dev/null 2>&1 \
       && ! pgrep -af "dyna_conv_v19d9.*gpu=${g}" > /dev/null 2>&1; then
      # better signal: check nvidia-smi for free memory >= 30GB
      local free
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null)
      if [ -n "$free" ] && [ "$free" -ge 30000 ]; then
        echo "$g"; return
      fi
    fi
  done
}

# Poll every 5 min for a freed GPU
echo "waiting for a GPU to free (>=30GB) for llava-1.5 N=1..."
while true; do
  gpu=$(free_gpu)
  if [ -n "$gpu" ]; then
    echo "GPU $gpu is free, launching llava-1.5 N=1"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python -u examiner/dyna_conv_v19d9.py \
      --dataset svg --num_samples 500 \
      --model_path llava-hf/llava-1.5-7b-hf \
      --num_contexts 1 \
      --outfile "$OUT" \
      --cache_file "$CACHE" \
      --max_rounds 20 \
      > "$LOG" 2>&1 &
    echo "launched llava-1.5 N=1 pid=$! gpu=$gpu -> $LOG"
    break
  fi
  sleep 300
done
