#!/bin/bash
set -euo pipefail

# Parallel runner for API-backed VLMs (v19).
# This mirrors the style of scripts/dyna-4o/final_run_all_parallel_full_resume5.sh:
# - one subshell per model
# - per-model log file
# - skip if output exists and log has no recent errors
# - wait for all jobs and return non-zero if any fails

usage() {
  cat <<'EOF'
Usage:
  ./scripts/dyna-v19/v19_api.sh [--max_rounds N] [--save_dir DIR] [--log_dir DIR] [--extra_args "..."]

Environment (export before running):
  - OPENAI_API_KEY   (OpenAI Responses API, vision)
  - GEMINI_API_KEY   (Google Gemini generateContent, vision)
  - ZHIPU_API_KEY    (Zhipu /chat/completions, vision)

Notes:
  - This script runs one model per provider in parallel.
  - MiniMax is NOT included by default because vision is not wired in infer/api_vlm.py yet.
  - Extra args are passed to examiner/dyna_conv_v19.py (e.g., --cache_file ...).

Examples:
  export OPENAI_API_KEY="..."
  export GEMINI_API_KEY="..."
  export ZHIPU_API_KEY="..."
  ./scripts/dyna-v19/v19_api.sh
  # All jobs always run with:
  #   --dataset vg --num_samples 100

EOF
}

# Locked to requested defaults
DATASET="vg"
NUM_SAMPLES="100"
MAX_ROUNDS="20"
SAVE_DIR="work_dirs/vg/v19_api"
LOG_DIR="work_dirs/logs_v19_api"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --max_rounds)
      MAX_ROUNDS="${2:-}"; shift 2 ;;
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

export PYTHONPATH="${PYTHONPATH:-}:./"
export PYTHONPATH="${PYTHONPATH:-}:/raid/william/project/context-eval-mllm/infer/LLaVA/llava"
export PYTHONPATH="/raid/william/project/context-eval-mllm/infer/LLaVA:${PYTHONPATH:-}"

RUN_FILE="examiner/dyna_conv_v19.py"

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

echo "Starting v19 API parallel inference..."
echo "  dataset     : ${DATASET}"
echo "  num_samples : ${NUM_SAMPLES}"
echo "  max_rounds  : ${MAX_ROUNDS}"
echo "  save_dir    : ${SAVE_DIR}"
echo "  log_dir     : ${LOG_DIR}"

run_job() {
  local name="$1"
  local model_path="$2"
  local required_env="$3"
  local outfile="${SAVE_DIR}/${name}.json"
  local logfile="${LOG_DIR}/${name}.log"

  (
    if [[ -z "${!required_env:-}" ]]; then
      echo "Missing ${required_env}. Skipping ${name} (${model_path})."
      exit 0
    fi

    echo "Running ${name}: ${model_path}"
    # Explicit cache file per model for safe resuming
    local cache_file="${outfile%.json}_cache.json"

    # Skip ONLY if we have evidence the previous run completed successfully.
    # If a run was interrupted, re-running is safe because dyna_conv_v19.py
    # will load the cache file and continue.
    if [[ -f "$outfile" ]] && [[ -f "$logfile" ]]; then
      if tail -n 80 "$logfile" | grep -iq "results saved to" \
        && tail -n 80 "$logfile" | grep -iq "completed processing" \
        && ! tail -n 80 "$logfile" | grep -iq "error\\|traceback\\|exception"; then
        echo "Skipping ${name} - previous run completed and log looks clean: ${outfile}"
        exit 0
      fi
    fi

    # API-only jobs: no GPU needed; use any env with openai+requests+PIL+nltk.
    # shellcheck disable=SC2086
    /raid/icy/iris/.conda/envs/coneval-haelm/bin/python "${RUN_FILE}" \
      --dataset "${DATASET}" \
      --num_samples "${NUM_SAMPLES}" \
      --max_rounds "${MAX_ROUNDS}" \
      --model_path "${model_path}" \
      --outfile "${outfile}" \
      --cache_file "${cache_file}" \
      ${EXTRA_ARGS}
  ) >"${logfile}" 2>&1 &
}

# All routed through uniapi gateway (OpenAI-compatible /chat/completions).
# parity/Harbor was retired — uniapi serves the same OpenAI + Gemini surface.
# - gpt-4o, gpt-5.4-mini are normal vision models on uniapi
# - gemini-2.5-flash is the regular vision-capable Gemini (NOT the image-gen
#   gemini-2.5-flash-image which produces base64 PNG bytes as "responses")
# - zhipu/glm-5v-turbo continues via Zhipu native endpoint (kept as-is for now)
run_job "openai_gpt-5.4-mini" "uniapi/gpt-5.4-mini" "UNIAPI_API_KEY"
run_job "openai_gpt-4o" "uniapi/gpt-4o" "UNIAPI_API_KEY"
run_job "gemini_gemini-2.5-flash" "uniapi/gemini-2.5-flash" "UNIAPI_API_KEY"
run_job "zhipu_glm-5v-turbo" "zhipu/glm-5v-turbo" "ZHIPU_API_KEY"

wait

echo "All v19 API VG jobs completed."
echo "VG outputs in: work_dirs/vg/v19_api"
# SVG block removed by user request — VG only.

