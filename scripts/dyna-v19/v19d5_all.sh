#!/bin/bash
set -uo pipefail

# Run v19.5 first batch (3 models) then second batch (3 models) sequentially.
# Required env: OPENAI_API_KEY (optional OPENAI_BASE_URL for a gateway), HF_TOKEN (for gemma-3-12b-it).

cd "$(dirname "$0")/../.."

bash scripts/dyna-v19/v19d5_qwen_gemma.sh
echo "---- first batch done at $(date +'%F %T') ----"
bash scripts/dyna-v19/v19d5_rest.sh
echo "---- second batch done at $(date +'%F %T') ----"
echo "All v19.5 runs complete."
