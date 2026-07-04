# coneval

A contextualized evaluation framework for Vision-Language Models (VLMs) that tests model faithfulness and hallucination through dynamic multi-turn conversations.

## Overview

This project implements a comprehensive evaluation pipeline that assesses whether VLMs hallucinate (produce responses inconsistent with image content) or remain faithful to the actual visual information. The evaluation uses multi-turn natural conversations guided by ground truth annotations from COCO and Visual Genome datasets.

## Environment Setup

```bash
# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install transformers and related libraries
pip install transformers
pip install accelerate
pip install bitsandbytes  # For 8-bit quantization (optional)

# Install vision and utility libraries
pip install pillow
pip install tqdm
pip install requests
pip install python-dotenv
pip install pydantic

# Install OpenAI SDK for LLM examiner
pip install openai
```

### 3. Install Model-Specific Dependencies

Depending on which VLMs you want to evaluate:

```bash
# For Qwen-VL models
pip install qwen-vl-utils

# For BLIP models
pip install salesforce-lavis

# For InternVL
pip install timm

# For scene graph processing
pip install networkx
```

### 4. Set Up API Keys

Create a `.env` file in the project root:

```bash
# OpenAI API (for GPT-4o examiner)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional: custom endpoint

# OR Azure OpenAI (alternative)
AZURE_OPENAI_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here
AZURE_OPENAI_DEPLOYNAME=gpt-4o  # Your deployment name
```

**Note**: The LLM examiner (GPT-4o) is required for dynamic conversation evaluation. You need either OpenAI API access or Azure OpenAI access.

### 5. Download Datasets

Run the provided script to download COCO dataset:

```bash
bash prepare.sh
```

This will download:
- COCO 2017 train/val/test images
- COCO 2014 train/val/test images
- Annotations for both versions

**Note**: This requires ~40GB of storage and may take several hours depending on your internet connection.

Alternatively, download manually:
```bash
mkdir -p data/coco && cd data/coco

# Download validation set (minimum required)
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract
unzip val2017.zip
unzip annotations_trainval2017.zip
```

For Visual Genome dataset, follow instructions in the Visual Genome website.

### 6. Set Python Path

Before running scripts, set the Python path:

```bash
export PYTHONPATH=$PYTHONPATH:./:infer:grader/easydetect
```

Or add to your `.bashrc`/`.zshrc`:
```bash
echo 'export PYTHONPATH=$PYTHONPATH:/path/to/context-eval-mllm' >> ~/.zshrc
source ~/.zshrc
```

### 7. Verify Installation

Test that imports work:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from openai import OpenAI; print('OpenAI SDK installed')"
```

### Resource Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 16GB | 24GB+ (40GB for large models) |
| RAM | 32GB | 64GB+ |
| Storage | 50GB | 100GB+ (with all datasets) |
| CPU | 8 cores | 16+ cores |

### Common Issues

**1. CUDA Out of Memory**
- Use smaller batch sizes
- Enable gradient checkpointing
- Use 8-bit quantization with `load_in_8bit=True`
- Try smaller model variants (e.g., 7B instead of 13B)

**2. Missing OpenAI API Key**
```
AssertionError: Model name is not provided.
```
- Ensure `.env` file exists with `OPENAI_API_KEY` or `AZURE_OPENAI_KEY`

**3. Import Errors**
```
ModuleNotFoundError: No module named 'examiner'
```
- Set `PYTHONPATH` correctly: `export PYTHONPATH=./`

**4. Model Download Issues**
- Models are downloaded from Hugging Face Hub automatically
- Requires ~10-30GB per model
- Set `HF_HOME` to control cache location: `export HF_HOME=/path/to/cache`

## Running the Evaluation Pipeline

### Quick Start with `scripts/run.sh`

The main script `context-eval-mllm/scripts/run.sh` provides a basic evaluation pipeline:

```bash
# Navigate to the project root
cd context-eval-mllm

# Run the evaluation pipeline
bash scripts/run.sh
```

**Script Location:** `context-eval-mllm/scripts/run.sh`

### What `scripts/run.sh` Does

The main evaluation pipeline consists of four key steps:

#### Step 1: Environment Setup
```bash
export PYTHONPATH=./
```
- Sets the Python path to enable proper module imports
- Ensures all custom modules (`examiner`, `infer`, `grader`, `utils`) can be imported

#### Step 2: Caption Generation
```bash
python examiner/caption.py --debug --outfile output/caption_200/caption.json
```
- **Purpose**: Generates baseline image descriptions using the VLM
- **Process**: 
  - Loads 200 sample images from the dataset
  - Asks VLM to provide detailed captions
  - Saves responses for later analysis
- **Output**: `output/caption_200/caption.json`
- **Note**: The script references `examiners/` but should be `examiner/` (singular)

#### Step 3: Dynamic Conversational Evaluation (JSON Mode)
```bash
python examiner/dyna_conv_json.py --debug \
    --p_mode coverage_certainty_with_answer \
    --outfile output/dyna_bad_examples_200/coverage_certainty_with_answer_json_mode.json
```
- **Purpose**: Conducts multi-turn conversations with structured output
- **Prompt Mode**: `coverage_certainty_with_answer`
  - **Coverage**: Questions cover all objects, attributes, and relations in the image
  - **Certainty**: Only asks questions with definite answers based on visible content
  - **With Answer**: Includes expected answers in structured JSON format
- **Process**: 
  - GPT-4o acts as an "examiner" generating questions
  - VLM responds to each question while viewing the image
  - Conversation continues until examiner outputs "END"
  - All exchanges are logged with structured metadata
- **Output**: `output/dyna_bad_examples_200/coverage_certainty_with_answer_json_mode.json`
- **Note**: File `dyna_conv_json.py` may not exist; might need to use `dyna_conv.py` instead

#### Step 4: Dynamic Conversational Evaluation (Standard Mode)
```bash
python examiner/dyna_conv.py --debug \
    --p_mode coverage_certainty \
    --outfile output/dyna_bad_examples_200/coverage_certainty.json
```
- **Purpose**: Same as Step 3 but with different output format
- **Prompt Mode**: `coverage_certainty` (without "_with_answer")
- **Difference**: More compact output format, similar evaluation logic
- **Output**: `output/dyna_bad_examples_200/coverage_certainty.json`

#### Step 5: Model Checkpoint Processing
```bash
CKPT=work_dirs/objectllama/sft/finetune_objectllama_full-ft_vllava-sft
CKPT_NAME=$(echo $CKPT | tr "/" "\n" | tail -1)
MODEL_BASE=${2: -None}
```
- **Purpose**: Sets up variables for fine-tuned model checkpoints
- **CKPT**: Full path to model checkpoint directory
- **CKPT_NAME**: Extracts just the checkpoint name (last directory)
- **MODEL_BASE**: Optional base model parameter (defaults to None)
- **Note**: Script appears incomplete here

### Alternative Scripts

The `scripts/` directory contains additional evaluation scripts:

#### Dynamic Conversation Scripts (`scripts/dyna/`)
- **`scripts/dyna/run.sh`**: Batch evaluation across multiple VLMs
  - Supports LLaVA, Qwen-VL, BLIP2, and more
  - Configurable sample sizes and output directories
  - Uses SLURM for cluster computing
- **`scripts/dyna/run_icl.sh`**: Evaluation with In-Context Learning examples
- **`scripts/dyna/llava.sh`**: LLaVA-specific evaluation

#### Caption Generation Scripts (`scripts/caption/`)
- Various scripts for generating captions with different models

#### Grading Scripts (`scripts/graders/`)
- Scripts to run different grading metrics (CHAIR, POPE, FaithScore, etc.)

### Customizing the Evaluation

To run your own evaluation, modify the script parameters:

```bash
# Set Python path
export PYTHONPATH=./

# Run dynamic conversation with custom settings
python examiner/dyna_conv.py \
    --dataset vg \                          # or 'coco'
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --p_mode coverage_certainty \           # evaluation strategy
    --num_samples 50 \                      # number of images
    --icls icls/vg_icl.json \              # optional: in-context examples
    --outfile output/my_eval/qwen_results.json
```

### Important Notes

⚠️ **Known Issues:**
1. The script references `examiners/` but the directory is `examiner/` (singular)
2. The file `dyna_conv_json.py` may not exist - use `dyna_conv.py` instead
3. Requires OpenAI API key set in `.env` file for GPT-4o examiner
4. Default model paths may need adjustment based on your setup

💡 **Tips:**
- Start with `--num_samples 10` for testing before full runs
- Use `--debug` flag to see detailed progress
- Check output directories exist or will be created automatically
- Monitor GPU memory usage with `nvidia-smi`

## Evaluation Strategies

The framework supports multiple evaluation modes via different prompt strategies:

### Coverage-Based
- **`coverage_certainty`**: Ensures all image content is questioned while maintaining answerable questions
- **`coverage_certainty_with_answer`**: Same as above with structured answer format

### Certainty-Based
- **`certainty`**: Only asks questions that can be confidently answered from the image
- Avoids ambiguous or speculative questions

### Unanswerable Questions
- **`unanswerable`**: Tests model's ability to refuse answering when:
  - Objects/attributes don't exist in the image
  - Information is unclear, hidden, or blurred
  - Questions ask about events before/after the captured moment
  - Background details not depicted in the image

### Adversarial Evaluation
- Tests with plausible but absent objects that commonly co-occur with visible ones
- Example: Asking about utensils when only a cake is visible

### POPE-Style Examiner (Dynamic Scene Graph)

The POPE-style examiner (`DSG.py`) extracts yes/no questions from VLM responses and evaluates object hallucination.

**Usage:**
```bash
python examiner/DSG.py \
    --conv_script <input_file> \
    --outfile <output_json_file_name> \
    --model_base <model_base> \
    --model_path <model_path> \
    --pope_model_name <pope_model_name> \
    --sample_num <number_of_samples> \
    --verbose
```

**Parameters:**
- **`--conv_script`**: Path to input conversational script file (JSON format with VLM responses)
  - Example: `output/caption/llava-1.5-7b-hf.json`
- **`--outfile`**: Output JSON file storing extracted questions and model answers
  - Example: `output/caption/pope/llava-1.5-7b-hf_pope.json`
- **`--model_base`**: Base model architecture (optional, default: None)
- **`--model_path`**: Path to the VLM being evaluated
  - Example: `liuhaotian/llava-v1.5-7b`
- **`--pope_model_name`**: GPT model used as the POPE question extractor
  - Default: `gpt-4o-mini`
  - Can also use `gpt-4o` for better quality
- **`--sample_num`**: Number of samples to process (default: 100)
- **`--verbose`**: Print extracted questions during processing

**How It Works:**
1. **Tuple Extraction**: GPT extracts semantic tuples from VLM responses
2. **Question Generation**: Converts tuples into natural language yes/no questions
3. **Re-evaluation**: Asks the VLM these extracted questions about the same image
4. **Scoring**: Evaluates consistency and hallucination based on yes/no answers

**Example:**
```bash
# Extract POPE questions from LLaVA-1.5-7B captions
python examiner/DSG.py \
    --conv_script output/caption/llava-1.5-7b-hf.json \
    --outfile output/caption/pope/llava-1.5-7b-hf_pope.json \
    --model_path liuhaotian/llava-v1.5-7b \
    --pope_model_name gpt-4o-mini \
    --sample_num 200 \
    --verbose
```

**Output Format:**
The output JSON contains:
- Original conversation data
- Extracted tuples per response
- Generated yes/no questions
- VLM answers to the questions
- Per-question and aggregate scores

## Sample Run Scripts

Here are complete, ready-to-run examples for the main evaluation scripts:

### 1. Context-Aware Dynamic Conversation (`dyna_conv_v6.py`)

The latest version (v6) supports contextualized evaluation with background and goals.

**Basic Usage:**
```bash
# Set Python path
export PYTHONPATH=./

# Run context-aware evaluation on COCO dataset
python examiner/dyna_conv_v6.py \
    --dataset coco \
    --num_samples 50 \
    --model_path liuhaotian/llava-v1.5-7b \
    --outfile output/dyna_v6/coco_llava_context.json
```

**Advanced Usage with Visual Genome:**
```bash
# Run on Visual Genome with Qwen-VL model
python examiner/dyna_conv_v6.py \
    --dataset vg \
    --num_samples 100 \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --outfile output/dyna_v6/vg_qwen_context.json \
    --cache_file output/dyna_v6/vg_qwen_cache.json
```

**With In-Context Learning Examples:**
```bash
# Use ICL examples to guide conversation style
python examiner/dyna_conv_v6.py \
    --dataset coco \
    --num_samples 200 \
    --model_path liuhaotian/llava-v1.5-13b \
    --icls icls/coco_icl.json \
    --outfile output/dyna_v6/coco_llava13b_icl.json
```

**Parameters for `dyna_conv_v6.py`:**
- `--dataset`: Dataset to use (`coco` or `vg`)
- `--num_samples`: Number of images to evaluate (default: 20)
- `--model_path`: HuggingFace model path or local path
- `--outfile`: Output JSON file path
- `--icls`: Optional path to in-context learning examples
- `--cache_file`: Cache file for resuming interrupted runs

**What Makes v6 Special:**
- **Context Generation**: Automatically generates realistic background and goal contexts
- **Node Selection**: Intelligently selects relevant objects based on context
- **Question Type Switching**: Dynamically switches between regular, follow-up, adversarial, and unanswerable questions
- **Resume Support**: Can resume from cache if interrupted

### 2. POPE-Style Evaluation (`DSG.py`)

Extract and evaluate yes/no questions from existing conversation outputs.

**Basic Usage:**
```bash
# Extract POPE questions from caption outputs
python examiner/DSG.py \
    --conv_script output/caption_200/caption.json \
    --outfile output/pope/caption_pope_eval.json \
    --model_path liuhaotian/llava-v1.5-7b \
    --sample_num 200
```

**With Different Models:**
```bash
# Evaluate Qwen-VL model
python examiner/DSG.py \
    --conv_script output/dyna/qwen_conversations.json \
    --outfile output/pope/qwen_pope_eval.json \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --pope_model_name gpt-4o \
    --sample_num 100 \
    --verbose
```

**With Model Base (for fine-tuned models):**
```bash
# Evaluate fine-tuned LLaVA variant
python examiner/DSG.py \
    --conv_script output/caption/finetuned_output.json \
    --outfile output/pope/finetuned_pope.json \
    --model_path ./checkpoints/llava-finetuned \
    --model_base liuhaotian/llava-v1.5-7b \
    --pope_model_name gpt-4o-mini \
    --sample_num 150
```

**Parameters for `DSG.py`:**
- `--conv_script`: Input JSON file with VLM responses
- `--outfile`: Output JSON file for POPE evaluation results
- `--model_path`: Path to the VLM being evaluated
- `--model_base`: Base model (for fine-tuned models, optional)
- `--pope_model_name`: GPT model for question extraction (default: `gpt-4o-mini`)
- `--sample_num`: Number of samples to process (default: 100)
- `--verbose`: Print extracted questions during processing

### 3. Complete Pipeline Example

Here's a complete workflow from caption generation to POPE evaluation:

```bash
#!/bin/bash
# Complete evaluation pipeline

# Set environment
export PYTHONPATH=./
MODEL_PATH="liuhaotian/llava-v1.5-7b"
DATASET="coco"
NUM_SAMPLES=100
OUTPUT_DIR="output/complete_eval"

mkdir -p $OUTPUT_DIR

# Step 1: Generate captions
echo "Step 1: Generating captions..."
python examiner/caption.py \
    --dataset $DATASET \
    --num_samples $NUM_SAMPLES \
    --model_path $MODEL_PATH \
    --outfile $OUTPUT_DIR/captions.json

# Step 2: Run context-aware dynamic conversation
echo "Step 2: Running context-aware evaluation..."
python examiner/dyna_conv_v6.py \
    --dataset $DATASET \
    --num_samples $NUM_SAMPLES \
    --model_path $MODEL_PATH \
    --outfile $OUTPUT_DIR/dyna_context.json

# Step 3: Extract POPE questions and re-evaluate
echo "Step 3: Extracting POPE questions..."
python examiner/DSG.py \
    --conv_script $OUTPUT_DIR/captions.json \
    --outfile $OUTPUT_DIR/pope_eval.json \
    --model_path $MODEL_PATH \
    --sample_num $NUM_SAMPLES \
    --verbose

echo "Evaluation complete! Results in $OUTPUT_DIR/"
```

### 4. Testing with Small Samples

Always test with small samples first:

```bash
# Quick test with 5 samples
python examiner/dyna_conv_v6.py \
    --dataset coco \
    --num_samples 5 \
    --model_path liuhaotian/llava-v1.5-7b \
    --outfile output/test/quick_test.json

# Verify output
cat output/test/quick_test.json | jq '.[0]' | head -20
```

### 5. Batch Evaluation Across Models

Evaluate multiple models in sequence:

```bash
#!/bin/bash
# Batch evaluation script

MODELS=(
    "liuhaotian/llava-v1.5-7b"
    "liuhaotian/llava-v1.5-13b"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "Salesforce/blip2-flan-t5-xl"
)

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo $MODEL | tr "/" "_")
    echo "Evaluating $MODEL_NAME..."
    
    python examiner/dyna_conv_v6.py \
        --dataset coco \
        --num_samples 50 \
        --model_path $MODEL \
        --outfile output/batch/$MODEL_NAME.json
done

echo "Batch evaluation complete!"
```

## Data Sources

The evaluation uses ground truth annotations from:

### COCO Dataset
- Multiple captions per image (5 captions)
- Object bounding boxes with coordinates `(x1, y1, x2, y2)` normalized to [0, 1]
- Located in `data/coco/val2017`

### Visual Genome Dataset
- Detailed scene graphs with:
  - Objects with attributes
  - Relationships between objects
  - Precise bounding box coordinates

## Key Components

### Examiner Module (`examiner/`)
- **`caption.py`**: Baseline caption generation
- **`dyna_conv.py`**: Dynamic conversation orchestration
- **`dyna_conv_v*.py`**: Various conversation strategy versions
- **`prompt.py`**: System prompts for different evaluation modes
- **`DSG.py`**: Dynamic Scene Graph utilities

### Inference Module (`infer/`)
- **`loader.py`**: Universal model loader supporting:
  - LLaVA (1.5, RLHF variants)
  - BLIP2, InstructBLIP
  - Qwen-VL (2, 2.5)
  - PaliGemma
  - Phi3.5-Vision, Phi4-Vision
  - InternVL, OPERA
  - And more
- Model-specific inference scripts for each VLM

### Grading Module (`grader/`)
Multiple evaluation metrics:
- **CHAIR**: Object hallucination metrics (COCO, Visual Genome)
- **POPE**: Polling-based object probing evaluation
- **FaithScore**: Faithfulness scoring
- **HaELM**: Hallucination evaluation
- **SPICE**: Scene graph-based evaluation
- **EasyDetect**: Automated hallucination detection

### Utilities (`utils/`)
- **`coco.py`**: COCO dataset utilities
- **`vg.py`**: Visual Genome utilities
- **`llm.py`**: LLM chat interface (GPT-4o)
- **`sg.py`**: Scene graph processing

## Conversational Evaluation Flow

1. **Ground Truth Loading**: Load image with annotations (objects, attributes, relations, bboxes)
2. **Context Formatting**: Convert annotations to structured text format
3. **LLM Examiner Setup**: Initialize GPT-4o with system prompt defining evaluation strategy
4. **Multi-turn Conversation**:
   - Examiner asks question based on image info and conversation history
   - VLM responds to the question with access to the actual image
   - Examiner evaluates response and asks follow-up questions
   - Process repeats until examiner outputs "END"
5. **Save Results**: Store full conversation history with prompts and responses

## In-Context Learning (ICL)

The framework supports few-shot learning via example conversations:
- **ICL files** in `icls/`:
  - `coco_icl.json`: COCO examples
  - `vg_icl.json`: Visual Genome examples
  - `*_uq.json`: Unanswerable question examples
- Loaded and prepended to conversations to guide the examiner's behavior

## Dataset Loaders

### Overview

The dataset loading system is located in the `utils/` directory and provides a unified interface for loading and formatting different datasets.

**Main Entry Point:** `utils/utils.py`

### Supported Datasets

#### 1. COCO Dataset (`utils/coco.py`)

**Loader Functions:**
- `load_coco2017(num_samples=None)`: Load COCO 2017 validation samples
- `load_sample_coco2017(img_id)`: Load a single COCO image by ID
- `format_case_coco(case)`: Format COCO data for LLM consumption

**Data Structure:**
```python
{
    "image_id": int,
    "image_url": str,
    "file_name": str,
    "instances": [
        {
            "category": str,          # Object category name
            "bbox": [x, y, w, h],     # Bounding box in pixels
            "pixel_area": float       # Object area in pixels
        },
        ...
    ],
    "captions": [str, ...]           # 5 human-written captions
}
```

**Formatted Output Example:**
```
descriptions:
A man sitting at a table with a laptop.
A person using a computer in an office.
...

instances:
person bbox: [100, 150, 200, 300] size: 60000
laptop bbox: [250, 200, 150, 100] size: 15000
```

#### 2. Visual Genome Dataset (`utils/vg.py`)

**Loader Functions:**
- `load_vg(num_samples=None)`: Load Visual Genome samples
- `load_sample_vg(idx)`: Load a single VG image by index
- `format_case_vg(case, use_region=False)`: Format VG data with scene graphs

**Data Structure:**
```python
{
    "image_id": int,
    "url": str,
    "width": int,
    "height": int,
    "image": PIL.Image,
    "sg": {
        "objects": {
            object_id: {
                "object_id": int,
                "names": [str],           # Object names
                "x": int, "y": int,       # Top-left corner
                "w": int, "h": int,       # Width, height
                "attributes": [str]       # Visual attributes
            },
            ...
        },
        "relationships": [
            {
                "subject": object_dict,    # Subject object
                "predicate": str,          # Relationship type
                "object": object_dict      # Object object
            },
            ...
        ],
        "regions": [...]                   # Region descriptions
    },
    "metadata": {...}                      # Original annotations
}
```

**Formatted Output Example:**
```
Instances:
instance 0, person, bbox: (0.10, 0.20, 0.35, 0.80), attributes: standing, smiling
instance 1, laptop, bbox: (0.40, 0.50, 0.65, 0.75), attributes: silver, open

Relation between the above instances:
person (instance 0) using laptop (instance 1)
person (instance 0) next to laptop (instance 1)
```

#### 3. Synthetic Visual Genome (SVG) Dataset (`utils/svg.py`)

**Pre-converted Dataset:** `Icey444/svg5000_in_vg`

For easier integration with the existing evaluation pipeline, we provide a pre-converted version of the SVG dataset that uses Visual Genome (VG) format. This eliminates the need for runtime format conversion and ensures compatibility with all VG-based evaluation tools.

**Key Features:**
- **Pre-converted to VG format**: All SVG samples converted to VG scene graph structure with `sg` key
- **Images included**: 5000 randomly sampled images embedded in the dataset (no separate download needed)
- **Ready to use**: Direct compatibility with `SceneGraphData.from_dict()`
- **Shuffled samples**: Random permutation for diverse evaluation sets

**Dataset Structure:**

The converted dataset maintains VG compatibility while preserving original SVG metadata:

```python
{
    "image_id": str,               # Original image filename
    "image": PIL.Image,            # Pre-loaded PIL image object
    "sg": {                        # VG-format scene graph
        "objects": {
            "0": {
                "object_id": 0,
                "x": int, "y": int, "w": int, "h": int,  # Pixel coordinates
                "names": [str],    # Object names
                "synsets": [],     # WordNet synsets (empty for SVG)
                "attributes": []   # Object attributes
            },
            ...
        },
        "relationships": [
            {
                "predicate": str,
                "synsets": [],
                "subject": {...},  # Full object dict
                "object": {...}    # Full object dict
            },
            ...
        ]
    },
    "width": int,                  # Image width in pixels
    "height": int,                 # Image height in pixels
    # Original SVG metadata preserved:
    "regions": [...],              # Original SVG regions
    "scene_graph": {...},          # Original SVG scene graph
    "url": str,                    # Original image URL
    "ann_types": [str]             # Annotation types
}
```

**Loader Functions:**
```python
# Load from pre-converted VG-format dataset
from utils.svg import load_svg, format_case_svg

# Load samples (automatically uses Icey444/svg5000_in_vg)
samples = load_svg(num_samples=100)

# Format for display (VG-style output)
formatted = format_case_svg(samples[0])
print(formatted)
```

**Formatted Output Example:**
```
Instances:
instance 0, bottle, bbox: (0.23, 0.70, 0.26, 0.87), attributes: none
instance 1, chair, bbox: (0.80, 0.68, 0.91, 0.88), attributes: black
instance 2, table, bbox: (0.72, 0.71, 1.00, 0.93), attributes: wooden

Relation between the above instances:
bottle (instance 0) on table (instance 2)
chair (instance 1) at table (instance 2)
```

**Usage in Evaluation:**

```python
# Run evaluation with pre-converted SVG dataset
python examiner/dyna_conv_v6.py \
    --dataset svg \
    --num_samples 20 \
    --model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --outfile output/dyna_conv_v6_svg.json
```

**No additional setup required** - images and scene graphs are loaded directly from HuggingFace.