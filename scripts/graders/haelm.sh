#!/usr/bin/env bash
#
# Run HaELM grader on a conversation JSON file.
#
# Usage (from project root):
#   ./scripts/graders/haelm.sh [options]
#   ./scripts/graders/haelm.sh --conv output/vg/icl.json --outfile output/vg/icl_haelm.json
#
# Options (override defaults):
#   --conv PATH         Input conversation JSON (default: output/vg/icl.json)
#   --llama_path PATH   LLaMA base model (default: checkpoints/llama-7b-hf)
#   --checkpoint_path   HaELM adapter checkpoint (default: graders/HaELM/checkpoint)
#   --outfile PATH      Output JSON (default: output/vg/icl_haelm.json)
#
# Example:
#   ./scripts/graders/haelm.sh --conv output/my_convs.json --outfile output/my_haelm.json

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python graders/HaELM/haelm.py "$@"
