#!/bin/bash
# Run VALOR object-existence grader on one transcript file via the local Qwen3 vLLM.
# Output: <input_dir>/<base>/valor/<base>_valor_obj_exist.json
#
# Usage:
#   bash scripts/graders/run_valor.sh <input.json> [--first-n N]
#
# Env overrides:
#   VALOR_OPENAI_BASE_URL (default: http://localhost:8088/v1)
#   VALOR_OPENAI_API_KEY  (default: william)
#   VALOR_MODEL           (default: Qwen/Qwen3-30B-A3B-Instruct-2507)
#   VALOR_PY              (default: /raid/miniconda3/envs/coneval-easydetect2/bin/python)
#
# Works for any transcript schema with sample["image_id"], sample["conversations"][*]["response"],
# and sample["sg"]["objects"] (or top-level sample["objects"]). All current examiners we run
# (CEDI v18/v19/v19conv/v19d91, caption, ablation_baseline2, caption_api, ablation_baseline2_api,
# v19_api) satisfy this.

set -u
INPUT="${1:?usage: run_valor.sh <input.json> [--first-n N]}"
shift || true
FIRST_N=99999
while [ $# -gt 0 ]; do
  case "$1" in
    --first-n) FIRST_N="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

ROOT_DIR=/raid/william/project/context-eval-mllm
cd "$ROOT_DIR"

BASE=$(basename "$INPUT" .json)
PARENT=$(dirname "$INPUT")
OUT_DIR="${PARENT}/${BASE}/valor"
OUT="${OUT_DIR}/${BASE}_valor_obj_exist.json"
mkdir -p "$OUT_DIR"

if [ -f "$OUT" ] && [ -s "$OUT" ]; then
  echo "[skip] $OUT already exists"
  exit 0
fi

VALOR_PY="${VALOR_PY:-/raid/miniconda3/envs/coneval-easydetect2/bin/python}"
export VALOR_OPENAI_BASE_URL="${VALOR_OPENAI_BASE_URL:-http://localhost:8088/v1}"
export VALOR_OPENAI_API_KEY="${VALOR_OPENAI_API_KEY:-william}"
export VALOR_MODEL="${VALOR_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
export PYTHONPATH="$ROOT_DIR"

echo "[valor] input    : $INPUT"
echo "[valor] output   : $OUT"
echo "[valor] backend  : $VALOR_OPENAI_BASE_URL  model=$VALOR_MODEL"
echo "[valor] first-n  : $FIRST_N"

exec "$VALOR_PY" graders/valor/evaluate_object_existence.py \
    -ip "$INPUT" -op "$OUT" --sample_num "$FIRST_N"
