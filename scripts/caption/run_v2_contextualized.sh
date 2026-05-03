#!/bin/bash
set -uo pipefail

# Contextualized caption runs for 4 models, pinned one-per-GPU on GPUs 0-3.
# Each run prefixes the caption query with the "first context" pulled from the
# matching v18 dyna run JSON for that same model.
#
# Models:
#   GPU 0 -> Qwen3-VL-8B-Instruct      (env: qwenvl3)
#   GPU 1 -> llava-1.5-7b-hf           (env: qwenvl3 — local llava env has stale torch 2.1)
#   GPU 2 -> InternVL3-8B-Instruct     (env: internvl)
#   GPU 3 -> opera-llava-1.5           (env: opera)

# Disable user-site to avoid /raid/icy/iris/.local leaking into env python paths.
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="./:infer:grader/easydetect"
export PYTHONPATH="${PYTHONPATH}:/raid/william/project/context-eval-mllm/infer/LLaVA"
export PYTHONPATH="${PYTHONPATH}:/raid/william/project/context-eval-mllm/infer/LLaVA/llava"

NUM_SAMPLES=100
DATASET=vg
SAVE_DIR=work_dirs/vg/caption_contextualized
LOG_DIR=work_dirs/logs_caption_contextualized
RUN_FILE=examiner/caption_contextualized.py

V18_DIR_RESUME5=work_dirs/vg/final_run_v18_gpt4o_resume5_completed
V18_DIR_COMPLETED=work_dirs/vg/final_run_v18_gpt4o_completed

mkdir -p "${SAVE_DIR}" "${LOG_DIR}" work_dirs/vg_image_cache

# Initialize conda
eval "$(conda shell.bash hook)"

run_job() {
    local name="$1"
    local conda_env="$2"
    local model_path="$3"
    local context_file="$4"
    local gpu_id="$5"

    local outfile="${SAVE_DIR}/${name}.json"
    local logfile="${LOG_DIR}/${name}.log"
    local cache_file="${outfile%.json}_cache.json"

    (
        if [[ ! -f "${context_file}" ]]; then
            echo "Missing context file ${context_file}; skipping ${name}." >&2
            exit 0
        fi

        if [[ -f "${outfile}" && -f "${logfile}" ]]; then
            if tail -n 80 "${logfile}" | grep -iq "Completed captions:" \
                && ! tail -n 80 "${logfile}" | grep -iqE "error|traceback|exception"; then
                echo "Skipping ${name} - previous run completed cleanly: ${outfile}"
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
            --cache_file "${cache_file}" \
            --context_file "${context_file}"
    ) >"${logfile}" 2>&1 &
}

# GPU 0 — Qwen3-VL-8B-Instruct
run_job "Qwen3-VL-8B-Instruct" \
    "work_dirs/envs/qwenvl3" \
    "Qwen/Qwen3-VL-8B-Instruct" \
    "${V18_DIR_RESUME5}/Qwen3-VL-8B-Instruct.json" \
    "0"

# GPU 1 — LLaVA-1.5-7B (using qwenvl3 env: has torch 2.9 + transformers 4.57)
run_job "llava-1.5-7b-hf" \
    "work_dirs/envs/qwenvl3" \
    "llava-hf/llava-1.5-7b-hf" \
    "${V18_DIR_RESUME5}/llava-1.5-7b-hf.json" \
    "1"

# GPU 2 — InternVL3-8B-Instruct
run_job "InternVL3-8B-Instruct" \
    "work_dirs/envs/internvl" \
    "OpenGVLab/InternVL3-8B-Instruct" \
    "${V18_DIR_RESUME5}/InternVL3-8B-Instruct.json" \
    "2"

# GPU 3 — Opera-LLaVA-1.5. v18 resume5 didn't complete for Opera, so we pull
# contexts from the earlier completed v18 run.
run_job "opera-llava-1.5" \
    "opera" \
    "/raid/william/project/context-eval-mllm/data/checkpoints/opera/llava-1.5" \
    "${V18_DIR_COMPLETED}/Opera-LLaVA-1.5.json" \
    "3"

wait

echo "All contextualized caption jobs done."
echo "Outputs in: ${SAVE_DIR}"
echo "Logs in:    ${LOG_DIR}"
