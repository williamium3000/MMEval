#!/bin/bash
set -uo pipefail

# v19.5 (no-gt) on VG for local VLMs, pinned to free GPUs in the 4-7 range.
# Examiner LLM is gpt-5/gpt-4o (per dyna_conv_v19d5.py).
#
# Required env vars (caller must export):
#   OPENAI_API_KEY   — OpenAI-compatible chat key
#   OPENAI_BASE_URL  — optional: override OpenAI endpoint (e.g. for a gateway)
#   HF_TOKEN         — required only for gated repos (gemma)

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY not set." >&2
    exit 2
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="./:infer:grader/easydetect"
export PYTHONPATH="${PYTHONPATH}:/raid/william/project/context-eval-mllm/infer/LLaVA"
export PYTHONPATH="${PYTHONPATH}:/raid/william/project/context-eval-mllm/infer/LLaVA/llava"

NUM_SAMPLES=100
DATASET=vg
SAVE_DIR=work_dirs/vg/v19d5
LOG_DIR=work_dirs/logs_v19d5
RUN_FILE=examiner/dyna_conv_v19d5.py

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

eval "$(conda shell.bash hook)"

run_job() {
    local name="$1"
    local conda_env="$2"
    local model_path="$3"
    local gpu_id="$4"

    local outfile="${SAVE_DIR}/${name}.json"
    local logfile="${LOG_DIR}/${name}.log"
    local cache_file="${outfile%.json}_cache.json"

    (
        if [[ -f "${outfile}" && -f "${logfile}" ]]; then
            if tail -n 80 "${logfile}" | grep -Eiq "Results saved to" \
                && ! tail -n 80 "${logfile}" | grep -Eiq "authentication|invalid token|无效的令牌"; then
                echo "Skipping ${name} - previous run looks clean: ${outfile}"
                exit 0
            fi
        fi

        conda activate "${conda_env}"
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        echo "[$(date +'%H:%M:%S')] Launching ${name} on GPU ${gpu_id} env=${conda_env} (${model_path})"
        python "${RUN_FILE}" \
            --dataset "${DATASET}" \
            --num_samples "${NUM_SAMPLES}" \
            --model_path "${model_path}" \
            --outfile "${outfile}" \
            --cache_file "${cache_file}"
    ) >"${logfile}" 2>&1 &
}

# GPU 4 — Qwen2.5-VL-7B-Instruct
run_job "Qwen2.5-VL-7B-Instruct" \
    "work_dirs/envs/qwenvl" \
    "Qwen/Qwen2.5-VL-7B-Instruct" \
    "4"

# GPU 6 — InternVL3-8B-Instruct
run_job "InternVL3-8B-Instruct" \
    "work_dirs/envs/internvl" \
    "OpenGVLab/InternVL3-8B-Instruct" \
    "6"

# GPU 7 — gemma-3-12b-it (gated on HF; requires HF_TOKEN env var)
run_job "gemma-3-12b-it" \
    "work_dirs/envs/gemma3" \
    "google/gemma-3-12b-it" \
    "7"

wait

echo "All v19.5 (no-gt) jobs done."
echo "Outputs in: ${SAVE_DIR}"
echo "Logs in:    ${LOG_DIR}"
