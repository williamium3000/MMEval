#!/bin/bash
set -euo pipefail

# Caption v2 - API models parallel runner (VG, 100 samples).
# Produces one caption per image (round_id=0) using examiner/caption.py.
# Includes per-model cache files so you can resume safely.

usage() {
  cat <<'EOF'
Usage:
  ./scripts/caption/run_v2_api.sh [--save_dir DIR] [--log_dir DIR] [--extra_args "..."]

Environment (export before running):
  - OPENAI_API_KEY
  - GEMINI_API_KEY
  - ZHIPU_API_KEY

Defaults (locked):
  --dataset vg
  --num_samples 100

EOF
}

SAVE_DIR="work_dirs/vg/caption_api"
LOG_DIR="work_dirs/logs_caption_api"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --save_dir)
      SAVE_DIR="${2:-}"; shift 2 ;;
    --log_dir)
      LOG_DIR="${2:-}"; shift 2 ;;
    --extra_args)
      EXTRA_ARGS="${2:-}"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

DATASET="vg"
NUM_SAMPLES="100"

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

export PYTHONPATH="${PYTHONPATH:-}:./:infer:grader/easydetect"

echo "Starting caption v2 (API) parallel jobs..."
echo "  dataset     : ${DATASET}"
echo "  num_samples : ${NUM_SAMPLES}"
echo "  save_dir    : ${SAVE_DIR}"
echo "  log_dir     : ${LOG_DIR}"

run_job() {
  local name="$1"
  local model_path="$2"
  local required_env="$3"
  local outfile="${SAVE_DIR}/${name}.json"
  local logfile="${LOG_DIR}/${name}.log"
  local cache_file="${outfile%.json}_cache.json"

  (
    if [[ -z "${!required_env:-}" ]]; then
      echo "Missing ${required_env}. Skipping ${name} (${model_path})."
      exit 0
    fi

    # Skip only if previous run completed cleanly
    if [[ -f "$outfile" ]] && [[ -f "$logfile" ]]; then
      if tail -n 80 "$logfile" | grep -iq "Completed captions:" \
        && ! tail -n 80 "$logfile" | grep -iq "error\\|traceback\\|exception"; then
        echo "Skipping ${name} - previous run completed and log looks clean: ${outfile}"
        exit 0
      fi
    fi

    echo "Running ${name}: ${model_path}"
    # shellcheck disable=SC2086
    /raid/icy/iris/.conda/envs/coneval-haelm/bin/python examiner/caption.py \
      --dataset "${DATASET}" \
      --num_samples "${NUM_SAMPLES}" \
      --model_path "${model_path}" \
      --outfile "${outfile}" \
      --cache_file "${cache_file}" \
      ${EXTRA_ARGS}
  ) >"${logfile}" 2>&1 &
}

# One newer model per provider (as of 2026 docs)
run_job "openai_gpt-5.4-mini" "uniapi/gpt-5.4-mini" "UNIAPI_API_KEY"
run_job "openai_gpt-4o" "uniapi/gpt-4o" "UNIAPI_API_KEY"
run_job "gemini_gemini-2.5-flash" "uniapi/gemini-2.5-flash" "UNIAPI_API_KEY"
run_job "zhipu_glm-5v-turbo" "zhipu/glm-5v-turbo" "ZHIPU_API_KEY"

wait

echo "All caption v2 API VG jobs completed."
echo "VG outputs in: work_dirs/vg/caption_api"
# SVG block removed by user request — VG only.

