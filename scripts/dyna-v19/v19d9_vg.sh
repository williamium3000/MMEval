#!/bin/bash
set -uo pipefail

# v19.9 on VG-100 — same as v19conv but the q_type switch prompt now demands a
# generally balanced distribution across regular / follow-up / adversarial /
# unanswerable (instead of just "be diverse"). Aimed at fixing the follow-up
# over-representation observed in v19conv runs.
#
# Usage:
#   bash scripts/dyna-v19/v19d9_vg.sh "<model>=<gpu>[,<model>=<gpu>...]"
#
# Example:
#   bash scripts/dyna-v19/v19d9_vg.sh \
#     "llava-1.5-7b-hf=0,InternVL3-8B-Instruct=1,gemma-3-12b-it=3"
#
# Required env:
#   OPENAI_API_KEY     uniapi key (also used for GEMINI_API_KEY when running the
#                      gemini-2.5-flash examinee)
#   OPENAI_BASE_URL    https://api.uniapi.io/v1
#   GEMINI_API_BASE    https://api.uniapi.io/gemini   (only needed for gemini)
#   HF_TOKEN           HuggingFace token (needed for gated repos like gemma-3-12b-it)

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY not set." >&2
    exit 2
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
# Make uniapi key fallback for gemini if user didn't set it explicitly
export GEMINI_API_KEY="${GEMINI_API_KEY:-${OPENAI_API_KEY}}"
export GEMINI_API_BASE="${GEMINI_API_BASE:-https://api.uniapi.io/gemini}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="./:infer:grader/easydetect"
export PYTHONPATH="${PYTHONPATH}:/raid/william/project/context-eval-mllm/infer/LLaVA"
export PYTHONPATH="${PYTHONPATH}:/raid/william/project/context-eval-mllm/infer/LLaVA/llava"

NUM_SAMPLES=100
DATASET=vg
RUN_FILE=examiner/dyna_conv_v19d9.py
SAVE_DIR=work_dirs/vg/v19d9
LOG_DIR=work_dirs/logs_v19d9_vg

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"
eval "$(conda shell.bash hook)"

run_job() {
    local name="$1" conda_env="$2" model_path="$3" gpu_id="$4"
    local outfile="${SAVE_DIR}/${name}.json"
    local logfile="${LOG_DIR}/${name}.log"
    local cache_file="${outfile%.json}_cache.json"

    (
        if [[ -f "${outfile}" && -f "${logfile}" ]]; then
            if tail -n 80 "${logfile}" | grep -Eiq "Results saved to" \
                && ! tail -n 80 "${logfile}" | grep -Eiqw "error|traceback|exception"; then
                echo "Skipping ${name} - previous run looks clean: ${outfile}"
                exit 0
            fi
        fi
        conda activate "${conda_env}"
        if [[ -n "${gpu_id}" ]]; then
            export CUDA_VISIBLE_DEVICES="${gpu_id}"
        else
            export CUDA_VISIBLE_DEVICES=""
        fi
        echo "[$(date +'%H:%M:%S')] Launching ${name} on GPU '${gpu_id:-cpu/api}' env=${conda_env}"
        python "${RUN_FILE}" \
            --dataset "${DATASET}" --num_samples "${NUM_SAMPLES}" \
            --model_path "${model_path}" \
            --outfile "${outfile}" --cache_file "${cache_file}"
    ) >"${logfile}" 2>&1 &
}

if [[ -z "${1:-}" ]]; then
    cat <<'EOF' >&2
Usage: bash v19d9_vg.sh "<model>=<gpu>[,<model>=<gpu>...]"
Models supported (use empty gpu id "" for API-only examinees):
  llava-1.5-7b-hf
  InternVL2-8B
  InternVL2_5-8B
  InternVL3-8B-Instruct
  Qwen2.5-VL-7B-Instruct
  gemma-3-12b-it
  opera-llava-1.5
  gemini-2.5-flash         (API; no GPU)

GPU policy: only use GPUs 0, 1, 3, 4 on this box (2/5/6/7 are reserved for
other users).
EOF
    exit 2
fi

declare -A model_path=(
    [llava-1.5-7b-hf]="llava-hf/llava-1.5-7b-hf:work_dirs/envs/qwenvl3"
    [InternVL2-8B]="OpenGVLab/InternVL2-8B:work_dirs/envs/internvl"
    [InternVL2_5-8B]="OpenGVLab/InternVL2_5-8B:work_dirs/envs/internvl"
    [InternVL3-8B-Instruct]="OpenGVLab/InternVL3-8B-Instruct:work_dirs/envs/internvl"
    [Qwen2.5-VL-7B-Instruct]="Qwen/Qwen2.5-VL-7B-Instruct:work_dirs/envs/qwenvl"
    [gemma-3-12b-it]="google/gemma-3-12b-it:work_dirs/envs/gemma3"
    [opera-llava-1.5]="/raid/william/project/context-eval-mllm/data/checkpoints/opera/llava-1.5:opera"
    [gemini-2.5-flash]="gemini/gemini-2.5-flash:work_dirs/envs/qwenvl3"
)

IFS=',' read -ra ASSIGN <<< "$1"
for pair in "${ASSIGN[@]}"; do
    name="${pair%=*}"; gpu="${pair#*=}"
    if [[ -z "${model_path[$name]:-}" ]]; then
        echo "unknown model: $name" >&2
        exit 2
    fi
    mp="${model_path[$name]%:*}"
    env="${model_path[$name]#*:}"
    run_job "$name" "$env" "$mp" "$gpu"
done
wait
echo "All v19.9 VG jobs in this batch done."
echo "Outputs: ${SAVE_DIR}"
echo "Logs:    ${LOG_DIR}"
