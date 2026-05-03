#!/bin/bash
set -uo pipefail

# Second batch of v19.5 (no-gt) runs. Intended to be launched AFTER
# scripts/dyna-v19/v19d5_qwen_gemma.sh finishes its three models.
#
# Models / GPU assignment:
#   GPU 4 -> llava-1.5-7b-hf       (env: qwenvl3 — local llava env has stale torch)
#   GPU 6 -> opera-llava-1.5       (env: opera)
#   GPU 7 -> InternVL2_5-8B        (env: internvl)
#
# Same env requirements as v19d5_qwen_gemma.sh: OPENAI_API_KEY (and optional
# OPENAI_BASE_URL to point at a gateway).

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

run_job "llava-1.5-7b-hf" \
    "work_dirs/envs/qwenvl3" \
    "llava-hf/llava-1.5-7b-hf" \
    "4"

run_job "opera-llava-1.5" \
    "opera" \
    "/raid/william/project/context-eval-mllm/data/checkpoints/opera/llava-1.5" \
    "6"

run_job "InternVL2_5-8B" \
    "work_dirs/envs/internvl" \
    "OpenGVLab/InternVL2_5-8B" \
    "7"

wait

echo "All v19.5 (no-gt) second-batch jobs done."
echo "Outputs in: ${SAVE_DIR}"
echo "Logs in:    ${LOG_DIR}"
