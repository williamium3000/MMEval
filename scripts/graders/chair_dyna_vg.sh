export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0

#!/bin/bash
# scripts/graders/chair.sh
eval "$(conda shell.bash hook)"
conda activate coneval-chair

# Export environment variables for subprocesses
export PYTHONPATH=./

# Parse arguments
USE_LOCAL=false
INPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            USE_LOCAL=true
            shift
            ;;
        *)
            if [ -z "$INPUT_DIR" ]; then
                INPUT_DIR="$1"
            fi
            shift
            ;;
    esac
done

# Set input directory (default if not provided)
INPUT_DIR="${INPUT_DIR:-work_dirs/vg/final_run_v18_gpt4o_conv_completed}"

# Load .env from project root for remote API URL
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Set API URL, key, and model based on --local flag
if [ "$USE_LOCAL" = true ]; then
    API_URL="http://localhost:8003/v1/chat/completions"
    API_KEY=""  # Local vLLM doesn't require API key
    MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
    echo "[$(date +'%H:%M:%S')] Using LOCAL vLLM server at $API_URL"
else
    API_URL="${REMOTE_API_URL}"
    API_KEY="${REMOTE_API_KEY}"
    MODEL="${REMOTE_API_MODEL:-Qwen3-30B-A3B-Instruct-2507}"
    echo "[$(date +'%H:%M:%S')] Using REMOTE API at $API_URL"
fi

# Function to process a single file
process_file() {
    model_file="$1"
    model_dir="$(dirname "$model_file")"
    
    # Check if hallucinated_words_*.json already exists in the same directory
    if ls "$model_dir"/hallucinated_words_*.json 1> /dev/null 2>&1; then
        echo "[$(date +'%H:%M:%S')] Skipping (already processed): $(basename $model_file)"
        return 0
    fi
    
    echo "[$(date +'%H:%M:%S')] Starting: $(basename $model_file)"
    
    if [ -z "$API_KEY" ]; then
        python grader/chair/chair_dyna_vg.py \
            "$model_file" \
            grader/chair/filtered_object_synsets_final.json \
            --api-url "$API_URL" \
            --model "$MODEL" \
            --group-by-image
    else
        python grader/chair/chair_dyna_vg.py \
            "$model_file" \
            grader/chair/filtered_object_synsets_final.json \
            --api-url "$API_URL" \
            --api-key "$API_KEY" \
            --model "$MODEL" \
            --group-by-image
    fi
    
    echo "[$(date +'%H:%M:%S')] Done: $(basename $model_file)"
}

# Export function and variables so they can be used by parallel processes
export -f process_file
export API_URL API_KEY MODEL USE_LOCAL

# Detect if this is a caption directory (files at depth 1)
# Check if there are JSON files directly in the input directory
IS_CAPTION_DIR=false
if find "$INPUT_DIR" -maxdepth 1 -name "*.json" -type f | grep -q .; then
    IS_CAPTION_DIR=true
    echo "[$(date +'%H:%M:%S')] Detected caption directory structure (depth 1)"
    
    # Organize files: copy non-hallucinated_words_ files into same-name folders
    echo "[$(date +'%H:%M:%S')] Organizing files into folders..."
    find "$INPUT_DIR" -maxdepth 1 -name "*.json" -type f | while read -r json_file; do
        filename=$(basename "$json_file")
        # Skip files that start with hallucinated_words_
        if [[ "$filename" == hallucinated_words_* ]]; then
            continue
        fi
        # Skip all *_pope_*.json (e.g. _pope_converted.json, _pope_output.json)
        if [[ "$filename" == *pope_* ]]; then
            continue
        fi
        # Skip all *_both_*.json (e.g. _with_both_answers.json)
        if [[ "$filename" == *both_* ]]; then
            continue
        fi
        
        # Extract folder name (remove .json extension)
        folder_name="${filename%.json}"
        folder_path="$INPUT_DIR/$folder_name"
        
        # Create folder if it doesn't exist
        if [ ! -d "$folder_path" ]; then
            mkdir -p "$folder_path"
        fi
        
        # Copy file into folder if it doesn't already exist there
        dest_file="$folder_path/$filename"
        if [ ! -f "$dest_file" ]; then
            cp "$json_file" "$dest_file"
            echo "[$(date +'%H:%M:%S')] Copied $filename -> $folder_name/"
        fi
    done
else
    echo "[$(date +'%H:%M:%S')] Detected nested directory structure (depth 2+)"
fi

# Process all model outputs in parallel (max 10 at a time)
# After organizing caption files, they're at depth 2, same as nested structures
# Filter out hallucinated_words_, *_pope_*.json, and *_both_*.json files
find "$INPUT_DIR" -mindepth 2 -maxdepth 2 -name "*.json" -type f | \
    grep -v '/hallucinated_words_' | \
    grep -v '_pope_' | \
    grep -v '_both_' | \
    xargs -n 1 -P 5 -I {} bash -c 'process_file "$@"' _ {}

echo "[$(date +'%H:%M:%S')] All files processed!"

# Collect all results into CSV
echo ""
echo "[$(date +'%H:%M:%S')] Collecting results into CSV..."
python utils/collect_results/chair.py \
    --input_dir "$INPUT_DIR"

echo "[$(date +'%H:%M:%S')] Done!"
