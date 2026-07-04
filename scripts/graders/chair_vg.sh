export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0

# python grader/chair/convert.py output/dyna_bad_examples/coverage_certainty.json
# python grader/chair/chair.py graders/chair/output/coverage_certainty.json

# python grader/chair/chair_vg.py $1 grader/chair/filtered_object_synsets_final.json

#!/bin/bash
# scripts/graders/chair.sh
eval "$(conda shell.bash hook)"
conda activate coneval-chair

# Export environment variables for subprocesses
export PYTHONPATH=./

# Set input directory (can be overridden by first argument)
INPUT_DIR="${1:-work_dirs/vg/caption}"
# Optional: only evaluate first n items per JSON (e.g. 100 for quick sanity check)
FIRST_N="${2:-}"

if [[ -n "$FIRST_N" ]]; then
    export FIRST_N
    echo "[$(date +'%H:%M:%S')] First-n mode: evaluating only first $FIRST_N items per file"
fi

# Function to process a single file (uses exported FIRST_N when set)
process_file() {
    model_file="$1"
    echo "[$(date +'%H:%M:%S')] Starting: $(basename $model_file)"
    first_n_args=()
    [[ -n "${FIRST_N:-}" ]] && first_n_args=(--first_n "$FIRST_N")
    python grader/chair/chair_vg.py \
        "$model_file" \
        grader/chair/filtered_object_synsets_final.json \
        "${first_n_args[@]}"
    echo "[$(date +'%H:%M:%S')] Done: $(basename $model_file)"
}

# Export function so it can be used by parallel processes
export -f process_file

# Process all model outputs in parallel (max 10 at a time)
# Using xargs with -P for parallel execution
# Exclude chair/mmhal output files so we only run on model caption JSONs
find "$INPUT_DIR" -name "*.json" -type f | grep -v -E 'hallucinated_words_|mmhal_' | \
    xargs -n 1 -P 10 -I {} bash -c 'process_file "$@"' _ {}

echo "[$(date +'%H:%M:%S')] All files processed!"

# Collect all results into CSV
echo ""
echo "[$(date +'%H:%M:%S')] Collecting results into CSV..."
python utils/collect_results/chair.py \
    --input_dir "$INPUT_DIR"

echo "[$(date +'%H:%M:%S')] Done!"
Done!"
