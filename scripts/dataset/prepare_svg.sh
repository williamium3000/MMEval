# set -e  # Exit on error

# echo "============================================================"
# echo "SVG Dataset Preparation Script"
# echo "============================================================"
# echo ""

# # Step 1: Download SVG annotations
# echo "Step 1/3: Downloading SVG annotations from HuggingFace..."
# echo "------------------------------------------------------------"
# python scripts/download_svg.py
# echo ""

# # Step 2: Download images
# echo "Step 2/3: Downloading images (ADE20K, COCO, Visual Genome)..."
# echo "------------------------------------------------------------"
# python scripts/download_svg_images.py
# echo ""

# Step 3: Create symlinks

# Step 3: Create symlinks
echo "Step 3/3: Creating unified image directory with symlinks..."
echo "------------------------------------------------------------"

OUTPUT_DIR="data/svg_images"
ALL_DIR="$OUTPUT_DIR/all"

# Clean and recreate all directory
echo "Cleaning $ALL_DIR..."
rm -rf "$ALL_DIR"
mkdir -p "$ALL_DIR"

# Convert ALL_DIR to absolute path
ALL_DIR=$(realpath "$ALL_DIR")

# Source directories
SOURCES=(
    "$OUTPUT_DIR/ade20k"
    "$OUTPUT_DIR/coco/train2017"
    "$OUTPUT_DIR/visual_genome/VG_100K"
    "$OUTPUT_DIR/visual_genome/VG_100K_2"
)

echo "Target directory: $ALL_DIR"
echo ""
echo "Creating symlinks from:"

for SOURCE in "${SOURCES[@]}"; do
    if [ ! -d "$SOURCE" ]; then
        echo "  ⚠ Skipping $SOURCE (does not exist)"
        continue
    fi
    
    # Convert to absolute path from current working directory
    SOURCE_ABS=$(realpath "$SOURCE")
    
    echo "  → $SOURCE_ABS"
    
    # Try wildcard symlink first (fast but may fail with too many files)
    if (cd "$ALL_DIR" && ln -s "$SOURCE_ABS"/* . 2>/dev/null); then
        echo "    ✓ Linked with wildcard"
    else
        echo "    Wildcard failed (too many files), moving files instead..."
        # Move files directly instead of symlinking
        find "$SOURCE_ABS" -maxdepth 1 -type f -exec mv {} "$ALL_DIR/" \; 2>/dev/null
        echo "    ✓ Moved files"
    fi
done

# Count total symlinks created
TOTAL_LINKED=$(find "$ALL_DIR" -type l | wc -l | tr -d ' ')

echo ""
echo "✓ Created $TOTAL_LINKED symlinks in $ALL_DIR"
echo ""