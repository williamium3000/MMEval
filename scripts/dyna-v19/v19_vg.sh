#!/bin/bash
set -uo pipefail

# v19 (the base, non-ban2type) on VG-100, for the 3 models we have ban2type
# baselines for: gemma-3-12b-it, llava-1.5-7b-hf, Qwen2.5-VL-7B-Instruct.
#
# Examiner LLM = Azure gpt-5.4 (via utils/llm.py's AzureOpenAI fallback) —
# apples-to-apples with the existing work_dirs/vg/v19ban2type/ results.
#
# Usage:
#   bash scripts/dyna-v19/v19_vg.sh "<model>=<gpu>[,<model>=<gpu>...]"

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" && -z "${AZURE_OPENAI_API_KEY:-}" && -z "${AZURE_OPENAI_KEY:-}" ]]; then
    echo "ERROR: neither OPENAI_API_KEY nor AZURE_OPENAI_API_KEY/AZURE_OPENAI_KEY is set." >&2
    exit 2
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
export GEMINI_API_KEY="${GEMINI_API_KEY:-${OPENAI_API_KEY:-}}"
export GEMINI_API_BASE="${GEMINI_API_BASE:-https://api.uniapi.io/gemini}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="./:infer:grader/easydetect"
export PYTHONPATH="${PYTHONPATH}:/raid/william/project/context-eval-mllm/infer/LLaVA"
export PYTHONPATH="${PYTHONPATH}:/raid/william/project/context-eval-mllm/infer/LLaVA/llava"

NUM_SAMPLES="${NUM_SAMPLES:-100}"
DATASET=vg
RUN_FILE=examiner/dyna_conv_v19.py
SAVE_DIR=work_dirs/vg/v19
LOG_DIR=work_dirs/logs_v19_vg
PARALLEL="${PARALLEL:-8}"

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

# Env dirs are directory-based (work_dirs/envs/{qwenvl3,internvl}); call the
# env's python directly rather than going through `conda activate` (no conda
# on this host).
env_python() {
    local env_dir="$1"
    if [[ -x "${env_dir}/bin/python" ]]; then
        echo "${env_dir}/bin/python"
    else
        echo "python3"
    fi
}

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
        if [[ -n "${gpu_id}" ]]; then
            export CUDA_VISIBLE_DEVICES="${gpu_id}"
        else
            export CUDA_VISIBLE_DEVICES=""
        fi
        local PY
        PY=$(env_python "${conda_env}")
        echo "[$(date +'%H:%M:%S')] Launching ${name} on GPU '${gpu_id:-cpu/api}' python=${PY}"
        "${PY}" "${RUN_FILE}" \
            --dataset "${DATASET}" --num_samples "${NUM_SAMPLES}" \
            --model_path "${model_path}" \
            --outfile "${outfile}" --cache_file "${cache_file}"
    ) >"${logfile}" 2>&1 &
}

if [[ -z "${1:-}" ]]; then
    cat <<'EOF' >&2
Usage: bash v19_vg.sh "<model>=<gpu>[,<model>=<gpu>...]"
Models supported (same registry as v19ban2type_vg.sh):
  llava-1.5-7b-hf
  InternVL2-8B
  InternVL2_5-8B
  InternVL3-8B-Instruct
  Qwen2.5-VL-7B-Instruct
  gemma-3-12b-it
  opera-llava-1.5
  gemini-2.5-flash         (API; no GPU)
EOF
    exit 2
fi

declare -A model_path=(
    [llava-1.5-7b-hf]="llava-hf/llava-1.5-7b-hf:work_dirs/envs/qwenvl3"
    [InternVL2-8B]="OpenGVLab/InternVL2-8B:work_dirs/envs/internvl"
    [InternVL2_5-8B]="OpenGVLab/InternVL2_5-8B:work_dirs/envs/internvl"
    [InternVL3-8B-Instruct]="OpenGVLab/InternVL3-8B-Instruct:work_dirs/envs/internvl"
    [Qwen2.5-VL-7B-Instruct]="Qwen/Qwen2.5-VL-7B-Instruct:work_dirs/envs/qwenvl3"
    [gemma-3-12b-it]="google/gemma-3-12b-it:work_dirs/envs/qwenvl3"
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
echo "All v19 VG jobs in this batch done."
echo "Outputs: ${SAVE_DIR}"
echo "Logs:    ${LOG_DIR}"
