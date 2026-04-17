#!/bin/bash
set -euo pipefail

# Parallel runner for API-backed VLMs (baseline2 ablation, dyna_conv).
# Mirrors the structure of:
# - scripts/caption/run_v2_api.sh
# - scripts/dyna-v19/v19_api.sh

usage() {
  cat <<'EOF'
Usage:
  ./scripts/dyna-4o-finalresume/ablation/final_run_all_parallel_baseline2_api.sh \
    [--num_samples N] [--max_rounds N] [--save_dir DIR] [--log_dir DIR] [--extra_args "..."]

Environment (export before running):
  - OPENAI_API_KEY or AZURE_OPENAI_KEY
      Used by examiner/dyna_conv.py for its internal "gpt-4o" evaluator.
  - OPENAI_API_KEY  (required if you want the OpenAI VLM job)
  - GEMINI_API_KEY  (required if you want the Gemini VLM job)
  - ZHIPU_API_KEY   (required if you want the Zhipu VLM job)

This script runs 3 API models in parallel (one per provider).
Outputs are cached per model so you can safely resume.
EOF
}

# Locked defaults (match the non-API baseline2 runner as closely as possible)
DATASET="vg"
NUM_SAMPLES="100"
MAX_ROUNDS="10"
P_MODE="CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL"

SAVE_DIR="work_dirs/vg/ablation_baseline2_api"
LOG_DIR="work_dirs/logs_ablation_baseline2_api"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --num_samples)
      NUM_SAMPLES="${2:-}"; shift 2 ;;
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

# dyna_conv.py's examiner uses utils/llm.LLMChat(model_name="gpt-4o"), which needs OpenAI or Azure.
if [[ -z "${OPENAI_API_KEY:-}" && -z "${AZURE_OPENAI_KEY:-}" ]]; then
  echo "Missing OPENAI_API_KEY or AZURE_OPENAI_KEY (required by examiner/dyna_conv.py)."
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:-}:./"
export PYTHONPATH="${PYTHONPATH:-}:/raid/william/project/context-eval-mllm/infer/LLaVA/llava"
export PYTHONPATH="/raid/william/project/context-eval-mllm/infer/LLaVA:${PYTHONPATH:-}"

RUN_FILE="examiner/dyna_conv.py"

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

echo "Starting baseline2 API parallel inference..."
echo "  dataset     : ${DATASET}"
echo "  num_samples : ${NUM_SAMPLES}"
echo "  max_rounds  : ${MAX_ROUNDS}"
echo "  p_mode      : ${P_MODE}"
echo "  save_dir    : ${SAVE_DIR}"
echo "  log_dir     : ${LOG_DIR}"

run_job() {
  local name="$1"
  local model_path="$2"
  local required_env_var="$3"

  local outfile="${SAVE_DIR}/${name}.json"
  local logfile="${LOG_DIR}/${name}.log"
  local cache_file="${outfile%.json}_cache.json"

  (
    if [[ -z "${!required_env_var:-}" ]]; then
      echo "Missing ${required_env_var}. Skipping ${name} (${model_path})."
      exit 0
    fi

    # Skip only if previous run completed cleanly.
    # (dyna_conv.py writes to stdout lines containing these substrings.)
    if [[ -f "$outfile" && -f "$logfile" ]]; then
      if tail -n 80 "$logfile" | grep -Eiq "Results saved to" \
        && tail -n 80 "$logfile" | grep -Eiq "Completed processing" \
        && ! tail -n 80 "$logfile" | grep -Eiq "error|traceback|exception"; then
        echo "Skipping ${name} - previous run completed and log looks clean: ${outfile}"
        exit 0
      fi
    fi

    echo "Running ${name}: ${model_path}"
    # shellcheck disable=SC2086
    python "${RUN_FILE}" \
      --dataset "${DATASET}" \
      --num_samples "${NUM_SAMPLES}" \
      --max_rounds "${MAX_ROUNDS}" \
      --p_mode "${P_MODE}" \
      --model_path "${model_path}" \
      --outfile "${outfile}" \
      --cache_file "${cache_file}" \
      ${EXTRA_ARGS}
  ) >"${logfile}" 2>&1 &
}

# One newer model per provider (as of 2026 docs)
run_job "openai_gpt-5.4-mini" "openai/gpt-5.4-mini" "OPENAI_API_KEY"
run_job "openai_gpt-4o" "openai/gpt-4o" "OPENAI_API_KEY"
run_job "gemini_gemini-2.5-flash-image" "gemini/gemini-2.5-flash-image" "GEMINI_API_KEY"
run_job "zhipu_glm-5v-turbo" "zhipu/glm-5v-turbo" "ZHIPU_API_KEY"

wait

echo "All baseline2 API jobs completed."
echo "Outputs in: ${SAVE_DIR}"
echo "Logs in   : ${LOG_DIR}"

