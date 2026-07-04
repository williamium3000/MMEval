#!/bin/bash
# bash scripts/graders/mmhal.sh --local work_dirs/vg/caption --first-n 100

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
unset all_proxy

source /raid/miniconda3/etc/profile.d/conda.sh
conda activate coneval-easydetect

# Maximum number of parallel jobs
MAX_JOBS=7

# Parse arguments
USE_LOCAL=false
FORCE=false
COLLECT_ONLY=false
COLLECT_BASE_DIR=""
INPUT_DIR=""
FIRST_N=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            USE_LOCAL=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --collect-only)
            COLLECT_ONLY=true
            shift
            ;;
        --first-n)
            FIRST_N="$2"
            shift 2
            ;;
        *)
            if [ -z "$INPUT_DIR" ]; then
                INPUT_DIR="$1"
            elif [ -z "$COLLECT_BASE_DIR" ]; then
                # Optional second positional arg for --collect-only mode
                COLLECT_BASE_DIR="$1"
            fi
            shift
            ;;
    esac
done

# Collect-only mode: generate mmhal CSVs for each subdir under work_dirs/vg (or provided base dir)
if [ "$COLLECT_ONLY" = true ]; then
    BASE_DIR="${COLLECT_BASE_DIR:-work_dirs/vg}"
    echo "📦 Collect-only mode enabled"
    echo "Base directory: $BASE_DIR"
    echo ""

    if [ ! -d "$BASE_DIR" ]; then
        echo "Error: base directory not found: $BASE_DIR"
        exit 1
    fi

    shopt -s nullglob
    subdirs=("$BASE_DIR"/*)
    shopt -u nullglob

    collected=0
    skipped=0
    for d in "${subdirs[@]}"; do
        [ ! -d "$d" ] && continue
        # Skip hidden dirs
        bn=$(basename "$d")
        [[ "$bn" == .* ]] && continue

        echo "--------------------------------------------------------------------------------"
        echo "Collecting MMHAL results in: $d"
        python utils/collect_results/mmhal.py --input_dir "$d" --output "mmhal_results_summary_${bn}.csv" || true
        collected=$((collected + 1))
    done

    echo ""
    echo "✅ Collect-only complete. Processed $collected subdir(s) under $BASE_DIR."
    exit 0
fi

# Set input directory (default if not provided)
INPUT_DIR="${INPUT_DIR:-work_dirs/vg/final_run_v18_gpt4o_completed}"

# Load .env from project root for remote API URL
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Set API URL and key based on --local flag
if [ "$USE_LOCAL" = true ]; then
    API_URL="http://localhost:8003/v1/chat/completions"
    API_KEY=""  # Local vLLM doesn't require API key
    echo "Using LOCAL vLLM server at $API_URL"
else
    API_URL="${REMOTE_API_URL}"
    API_KEY="${REMOTE_API_KEY}"
    if [ -z "$API_URL" ] || [ -z "$API_KEY" ]; then
        echo "Error: REMOTE_API_URL and REMOTE_API_KEY must be set in .env when not using --local"
        exit 1
    fi
    echo "Using REMOTE API at $API_URL"
fi

# Get all model files into array
mapfile -t model_files < <(find "$INPUT_DIR" -maxdepth 1 -name "*.json" -type f)
total_files=${#model_files[@]}

echo "Input directory: $INPUT_DIR"
echo "Found $total_files model files to evaluate"
[ -n "$FIRST_N" ] && echo "Limiting to first $FIRST_N items per JSON (--first-n $FIRST_N)"
echo "Running with $MAX_JOBS parallel jobs"
echo "========================================"
echo ""

# Counter for completed jobs
completed=0
skipped=0
# Array to track running PIDs
declare -a running_pids=()

for model_file in "${model_files[@]}"; do
    dir=$(dirname "$model_file")
    base=$(basename "$model_file" .json)
    
    # Skip files starting with hallucinated_words_
    if [[ "$base" == hallucinated_words_* ]]; then
        echo "⏭️  Skipping (hallucinated_words): $base"
        ((skipped++))
        continue
    fi
    
    # Skip all *_pope_*.json (e.g. _pope_converted.json, _pope_output.json)
    if [[ "$base" == *pope_* ]]; then
        echo "⏭️  Skipping (pope): $base"
        ((skipped++))
        continue
    fi

    # Skip all *_both_*.json (e.g. _with_both_answers.json)
    if [[ "$base" == *both_* ]]; then
        echo "⏭️  Skipping (both): $base"
        ((skipped++))
        continue
    fi
    
    # Caption dir: use raw file + --gt-type caption (no conversion; gt from image info like examiner)
    input_file="$model_file"
    gt_type_arg=""
    if [[ "$INPUT_DIR" == *"caption"* ]]; then
        gt_type_arg="--gt-type caption"
        # No conversion: caption JSON already has conversations (prompt/response) and sg/metadata for gt
    fi

    first_n_arg=""
    if [ -n "$FIRST_N" ]; then
        first_n_arg="--first-n $FIRST_N"
    fi
    
    eval_file="${dir}/${base}/mmhal_${base}.json"
    log_file="${dir}/${base}/mmhal_${base}.log"

    # Skip if evaluation already exists (unless --force is used)
    if [ -f "$eval_file" ] && [ "$FORCE" = false ]; then
        echo "⏭️  Skipping (exists): $base"
        ((skipped++))
        continue
    fi

    # Create output directory if it doesn't exist
    mkdir -p "${dir}/${base}"

    # Wait if we have MAX_JOBS running - check and clean up finished PIDs
    while [ ${#running_pids[@]} -ge $MAX_JOBS ]; do
        # Check which PIDs are still running
        new_pids=()
        for pid in "${running_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            else
                wait "$pid" 2>/dev/null
                ((completed++))
            fi
        done
        running_pids=("${new_pids[@]}")
        sleep 0.5
    done

    echo "🚀 [$((completed + ${#running_pids[@]} + skipped))/$total_files] Starting: $base"

    # Run in background
    (
        if [ "$USE_LOCAL" = true ]; then
            # Local vLLM - use Qwen3-30B model name (with Qwen/ prefix) and no API key needed
            if [ -z "$API_KEY" ]; then
                python grader/mmhal/mmhal_grader.py \
                    --response "$input_file" \
                    --evaluation "$eval_file" \
                    --gpt-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
                    --api-url "$API_URL" \
                    $gt_type_arg $first_n_arg && \
                echo "✅ Completed: $base"
            else
                python grader/mmhal/mmhal_grader.py \
                    --response "$input_file" \
                    --evaluation "$eval_file" \
                    --gpt-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
                    --api-url "$API_URL" \
                    --api-key "$API_KEY" \
                    $gt_type_arg $first_n_arg && \
                echo "✅ Completed: $base"
            fi
        else
            # Remote API (uses REMOTE_API_URL and REMOTE_API_KEY from .env)
            python grader/mmhal/mmhal_grader.py \
                --response "$input_file" \
                --evaluation "$eval_file" \
                --gpt-model Llama-3.1-70B-Instruct \
                --api-url "$API_URL" \
                --api-key "$API_KEY" \
                $gt_type_arg $first_n_arg && \
            echo "✅ Completed: $base"
        fi
    ) > "$log_file" 2>&1 &
    
    # Track the PID (don't increment completed here - wait for job to finish)
    running_pids+=($!)
done

# Wait for all remaining background jobs to complete
if [ ${#running_pids[@]} -gt 0 ]; then
    echo ""
    echo "[$(date +'%H:%M:%S')] Waiting for ${#running_pids[@]} remaining jobs to complete..."
    for pid in "${running_pids[@]}"; do
        if wait "$pid" 2>/dev/null; then
            ((completed++))
        else
            # Job failed, but still count it as attempted
            ((completed++))
        fi
    done
    echo "[$(date +'%H:%M:%S')] All jobs completed!"
fi

echo ""
echo "========================================"
echo "Summary:"
echo "  Evaluated: $completed"
echo "  Skipped:   $skipped"
echo "  Total:     $total_files"
echo "========================================"
echo ""
echo "💾 Run collect script to generate CSV:"
echo "   bash scripts/graders/collect_mmhal_results.sh"