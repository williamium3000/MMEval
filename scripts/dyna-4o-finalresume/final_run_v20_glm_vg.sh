#!/bin/bash

# v20 VG: GLM-5V-Turbo (Zhipu API).
# Single API model, no GPU required. Requires ZHIPU_API_KEY.

export PYTHONPATH=$PYTHONPATH:./
export PYTHONPATH=$PYTHONPATH:/raid/william/project/context-eval-mllm/infer/LLaVA/llava
export PYTHONPATH="/raid/william/project/context-eval-mllm/infer/LLaVA:${PYTHONPATH:-}"

NUM_SAMPLES=100
SAVE_DIR=work_dirs/vg/final_run_v20_gpt4o_small
RUN_FILE=examiner/dyna_conv_v20.py
LOG_DIR=work_dirs/logs_parallel_v20_small

mkdir -p ${SAVE_DIR} ${LOG_DIR}

eval "$(conda shell.bash hook)"

if [ -z "${ZHIPU_API_KEY:-}" ]; then
  echo "ERROR: ZHIPU_API_KEY is not set. Export it first." >&2
  exit 1
fi

echo "Starting v20 GLM-5V-Turbo VG inference..."
echo "  save_dir: ${SAVE_DIR}"
echo "  log_dir:  ${LOG_DIR}"

OUTFILE=$SAVE_DIR/zhipu_glm-5v-turbo.json
LOGFILE=${LOG_DIR}/zhipu_glm-5v-turbo.log
if [ -f "$OUTFILE" ] && [ -f "$LOGFILE" ] && ! tail -n 5 "$LOGFILE" | grep -iq "error"; then
  echo "Skipping zhipu_glm-5v-turbo - output exists and no errors: $OUTFILE"
else
  # Use the same env as the OpenAI examiner — no conda env switch needed.
  # Pick any conda env that has openai+requests installed; internvl or qwenvl works.
  conda activate work_dirs/envs/internvl
  python $RUN_FILE --dataset vg --model_path zhipu/glm-5v-turbo \
    --outfile $OUTFILE --num_samples $NUM_SAMPLES > "$LOGFILE" 2>&1
fi

echo "v20 GLM job completed!"
echo "Output: $OUTFILE"
echo "Log:    $LOGFILE"
