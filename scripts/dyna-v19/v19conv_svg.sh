#!/bin/bash
set -uo pipefail

# v19conv on SVG-500 for the 6 main local models (opera handled separately by VG resume).
# v19conv = conversation-aware variant of v19 (multi-turn examinee memory across rounds).
# Examiner LLMs: gpt-5.4-2026-03-05 for both context and conv (Azure modelhub).

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

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="./:infer:grader/easydetect"
export PYTHONPATH="${PYTHONPATH}:/raid/william/project/context-eval-mllm/infer/LLaVA"
export PYTHONPATH="${PYTHONPATH}:/raid/william/project/context-eval-mllm/infer/LLaVA/llava"

NUM_SAMPLES="${NUM_SAMPLES:-500}"
DATASET=svg
RUN_FILE=examiner/dyna_conv_v19conv.py
SAVE_DIR=work_dirs/svg/v19conv
LOG_DIR=work_dirs/logs_v19conv_svg
# Sample-level concurrency. Local VLM serialized through a lock; API calls
# fan out. PARALLEL=4 is a memory-conservative default after OOM at 8.
PARALLEL="${PARALLEL:-4}"

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
        # opera env needs OPERA's forked transformers-4.29.2 (adds
        # opera_decoding + related kwargs to GenerationMixin.generate).
        # The vanilla env install pth is broken; we cloned OPERA's source
        # to /raid/icy/iris/opera_deps/OPERA-src and installed editably.
        # PYTHONNOUSERSITE=1 disables .pth processing, so prepend the real
        # src dir directly to PYTHONPATH for the opera env only.
        if [[ "${conda_env}" == "opera" ]]; then
            export PYTHONPATH="/raid/icy/iris/opera_deps/OPERA-src/transformers-4.29.2/src:${PYTHONPATH}"
        fi
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        echo "[$(date +'%H:%M:%S')] Launching ${name} on GPU ${gpu_id} env=${conda_env}"
        python "${RUN_FILE}" \
            --dataset "${DATASET}" --num_samples "${NUM_SAMPLES}" \
            --model_path "${model_path}" \
            --outfile "${outfile}" --cache_file "${cache_file}" \
            --parallel "${PARALLEL}"
    ) >"${logfile}" 2>&1 &
}

# Argument: comma-separated list of (model_name=GPU_id) pairs to launch.
# Usage: bash v19conv_svg.sh "llava-1.5-7b-hf=7,InternVL3-8B-Instruct=6,gemma-3-12b-it=2"
if [[ -z "${1:-}" ]]; then
    echo "Usage: $0 <comma-separated model=gpu pairs>"
    echo "  models: llava-1.5-7b-hf, InternVL2-8B, InternVL2_5-8B, InternVL3-8B-Instruct, Qwen2.5-VL-7B-Instruct, gemma-3-12b-it"
    exit 2
fi

declare -A model_path=(
    [llava-1.5-7b-hf]="llava-hf/llava-1.5-7b-hf:work_dirs/envs/qwenvl3"
    [InternVL2-8B]="OpenGVLab/InternVL2-8B:work_dirs/envs/internvl"
    [InternVL2_5-8B]="OpenGVLab/InternVL2_5-8B:work_dirs/envs/internvl"
    [InternVL3-8B-Instruct]="OpenGVLab/InternVL3-8B-Instruct:work_dirs/envs/internvl"
    [Qwen2.5-VL-3B-Instruct]="Qwen/Qwen2.5-VL-3B-Instruct:work_dirs/envs/qwenvl3"
    [Qwen2.5-VL-7B-Instruct]="Qwen/Qwen2.5-VL-7B-Instruct:work_dirs/envs/qwenvl3"
    [Qwen2.5-VL-32B-Instruct]="Qwen/Qwen2.5-VL-32B-Instruct:work_dirs/envs/qwenvl3"
    [gemma-3-12b-it]="google/gemma-3-12b-it:work_dirs/envs/qwenvl3"
    [opera-llava-1.5]="/raid/william/project/context-eval-mllm/data/checkpoints/opera/llava-1.5:opera"
    # API examinees (route via gemini/<model> prefix; needs GEMINI_API_KEY+GEMINI_API_BASE)
    [gemini-2.5-flash]="gemini/gemini-2.5-flash:work_dirs/envs/qwenvl3"
)

IFS=',' read -ra ASSIGN <<< "$1"
for pair in "${ASSIGN[@]}"; do
    name="${pair%=*}"; gpu="${pair#*=}"
    if [[ -z "${model_path[$name]:-}" ]]; then echo "unknown model: $name" >&2; exit 2; fi
    mp="${model_path[$name]%:*}"
    env="${model_path[$name]#*:}"
    run_job "$name" "$env" "$mp" "$gpu"
done
wait
echo "All v19conv SVG jobs in this batch done."
