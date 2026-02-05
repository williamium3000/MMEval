#!/bin/bash
# scripts/graders/convert_captions.sh
# Convert all caption JSON files using convert_caption.py

eval "$(conda shell.bash hook)"
conda activate coneval-chair

# Export environment variables for subprocesses
export PYTHONPATH=./

# Set input directory
INPUT_DIR="${1:-work_dirs/vg/caption}"

echo "[$(date +'%H:%M:%S')] Converting caption files in: $INPUT_DIR"
echo ""

# Counter for processed files
processed=0
skipped=0

# Process each JSON file
for json_file in "$INPUT_DIR"/*.json; do
    # Skip if no files found
    [ -e "$json_file" ] || continue
    
    filename=$(basename "$json_file")
    converted_filename="${filename%.json}_converted-mmal.json"
    converted_path="$INPUT_DIR/$converted_filename"
    
    # Check if converted file already exists
    if [ -f "$converted_path" ]; then
        echo "[$(date +'%H:%M:%S')] Skipping (already converted): $filename"
        ((skipped++))
        continue
    fi
    
    echo "[$(date +'%H:%M:%S')] Converting: $filename"
    python grader/mmhal/convert_caption.py "$json_file"
    
    if [ $? -eq 0 ]; then
        echo "[$(date +'%H:%M:%S')] ✓ Done: $converted_filename"
        ((processed++))
    else
        echo "[$(date +'%H:%M:%S')] ✗ Failed: $filename"
    fi
    echo ""
done

echo "================================================================"
echo "[$(date +'%H:%M:%S')] Summary:"
echo "  Processed: $processed files"
echo "  Skipped: $skipped files"
echo "[$(date +'%H:%M:%S')] All done!"
