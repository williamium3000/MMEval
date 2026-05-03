#!/bin/bash
set -uo pipefail

# v19 (with gt) on SVG-500 for the 7 main models.
# Examiner LLM: gpt-5 (context) + gpt-4o (conversation) via OpenAI-compatible API.
#
# Required env: OPENAI_API_KEY (optional OPENAI_BASE_URL for a gateway), HF_TOKEN (gemma).
#
# GPU layout (concurrent — 7 SVG jobs on GPUs 0/1/2/3/4/6/7):
#   GPU 0  -> SVG llava-1.5-7b-hf       (env: qwenvl3 — local llava env has stale torch)
#   GPU 1  -> SVG InternVL2-8B          (env: internvl)
#   GPU 2  -> SVG InternVL2_5-8B        (env: internvl)
#   GPU 3  -> SVG InternVL3-8B-Instruct (env: internvl)
#   GPU 4  -> SVG Qwen2.5-VL-7B-Instruct (env: qwenvl)
#   GPU 6  -> SVG gemma-3-12b-it        (env: gemma3)
#   GPU 7  -> SVG opera-llava-1.5       (env: opera)  -- this is the only v19 run
#                                                       that fills a missing-v18 gap
#                                                       (SVG opera had no baseline2 run).

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

SVG_NUM_SAMPLES=500
RUN_FILE=examiner/dyna_conv_v19.py
MAX_ROUNDS=20

SVG_SAVE_DIR=work_dirs/svg/v19
SVG_LOG_DIR=work_dirs/logs_v19_svg

mkdir -p "${SVG_SAVE_DIR}" "${SVG_LOG_DIR}"

eval "$(conda shell.bash hook)"

run_job() {
    local name="$1"
    local conda_env="$2"
    local model_path="$3"
    local gpu_id="$4"
    local dataset="$5"     # "svg" or "vg"
    local num_samples="$6"
    local save_dir="$7"
    local log_dir="$8"
    local outfile="${save_dir}/${name}.json"
    local logfile="${log_dir}/${name}.log"
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
        echo "[$(date +'%H:%M:%S')] Launching ${name} (${dataset}/${num_samples}) on GPU ${gpu_id} env=${conda_env}"
        python "${RUN_FILE}" \
            --dataset "${dataset}" \
            --num_samples "${num_samples}" \
            --max_rounds "${MAX_ROUNDS}" \
            --model_path "${model_path}" \
            --outfile "${outfile}" \
            --cache_file "${cache_file}"
    ) >"${logfile}" 2>&1 &
}

# ── SVG-500 runs (7 models, parallel on GPUs 0/1/2/3/4/6/7) ───────────────────

run_job "llava-1.5-7b-hf" \
    "work_dirs/envs/qwenvl3" "llava-hf/llava-1.5-7b-hf" "0" \
    "svg" "${SVG_NUM_SAMPLES}" "${SVG_SAVE_DIR}" "${SVG_LOG_DIR}"

run_job "InternVL2-8B" \
    "work_dirs/envs/internvl" "OpenGVLab/InternVL2-8B" "1" \
    "svg" "${SVG_NUM_SAMPLES}" "${SVG_SAVE_DIR}" "${SVG_LOG_DIR}"

run_job "InternVL2_5-8B" \
    "work_dirs/envs/internvl" "OpenGVLab/InternVL2_5-8B" "2" \
    "svg" "${SVG_NUM_SAMPLES}" "${SVG_SAVE_DIR}" "${SVG_LOG_DIR}"

run_job "InternVL3-8B-Instruct" \
    "work_dirs/envs/internvl" "OpenGVLab/InternVL3-8B-Instruct" "3" \
    "svg" "${SVG_NUM_SAMPLES}" "${SVG_SAVE_DIR}" "${SVG_LOG_DIR}"

run_job "Qwen2.5-VL-7B-Instruct" \
    "work_dirs/envs/qwenvl" "Qwen/Qwen2.5-VL-7B-Instruct" "4" \
    "svg" "${SVG_NUM_SAMPLES}" "${SVG_SAVE_DIR}" "${SVG_LOG_DIR}"

run_job "gemma-3-12b-it" \
    "work_dirs/envs/gemma3" "google/gemma-3-12b-it" "6" \
    "svg" "${SVG_NUM_SAMPLES}" "${SVG_SAVE_DIR}" "${SVG_LOG_DIR}"

run_job "opera-llava-1.5" \
    "opera" "/raid/william/project/context-eval-mllm/data/checkpoints/opera/llava-1.5" "7" \
    "svg" "${SVG_NUM_SAMPLES}" "${SVG_SAVE_DIR}" "${SVG_LOG_DIR}"

wait

echo "All v19 SVG-500 jobs done."
echo "SVG outputs: ${SVG_SAVE_DIR}"
