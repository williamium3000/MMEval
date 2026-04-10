#!/bin/bash
# Simple inference test for infer/infer_qwenvl3.py with Qwen3-VL-8B-Instruct
# Uses conda env qwenvl3 and GPUs 1,2

set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-.}:./infer"

CONDA_ENV="qwenvl3"
# Prefer project env if present
[ -d "/raid/ztw/envs/qwen3_vl" ] && CONDA_ENV="/raid/ztw/envs/qwen3_vl"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export CUDA_VISIBLE_DEVICES=1,2

INFILE="${1:-tmp/test_qwenvl3_simple.json}"
OUTFILE="${2:-tmp/test_qwenvl3_simple_out.json}"
mkdir -p tmp

echo "Env: $CONDA_ENV, GPUs: $CUDA_VISIBLE_DEVICES"
echo "Infile: $INFILE"
echo "Outfile: $OUTFILE"
echo "Model: Qwen/Qwen3-VL-8B-Instruct"
echo ""

python infer/infer_qwenvl3.py \
  --infile "$INFILE" \
  --outfile "$OUTFILE" \
  --img_dir "" \
  --model_path Qwen/Qwen3-VL-8B-Instruct

echo ""
echo "Done. Output: $OUTFILE"
cat "$OUTFILE"
