#!/usr/bin/env bash
#
# Run FaithScore grader on a conversation JSON file.
#
# Usage (from project root):
#   export OPENAI_API_KEY="your-key"   # required for eval mode unless using --use_llama
#   ./scripts/graders/faithscore.sh [options]
#
# Options (override defaults):
#   --conv PATH         Input conversation JSON (default: output/vg/caption.json)
#   --vem_type TYPE     VEM: ofa-ve, ofa, or llava (default: llava)
#   --llava_path PATH   LLaVA model path when vem_type=llava (default: checkpoints/llava-v1.5-7b)
#   --openai_model NAME OpenAI model for stages (default: gpt-5)
#   --use_llama         Use local LLaMA instead of OpenAI (requires --llama_path)
#   --llama_path PATH   LLaMA model path when --use_llama
#   --sample_num N      Number of samples to evaluate (default: 100)
#   --start_idx N       Start index (default: 0)
#   --save_judgments P  Save LLM judgments to file P
#
# Modes:
#   --mode eval         Run evaluation (default)
#   --mode merge        Merge judgment files (use with --judgment_files, --merge_output)
#   --mode recalculate  Recompute scores from judgments (use with --judgment_file)
#
# Example (single file, OpenAI):
#   ./scripts/graders/faithscore.sh --conv output/vg/caption.json --sample_num 50
#
# Example (local LLaMA):
#   ./scripts/graders/faithscore.sh --conv output/vg/caption.json --use_llama --llama_path checkpoints/llama-7b-hf

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python graders/faithscore/eval.py "$@"
