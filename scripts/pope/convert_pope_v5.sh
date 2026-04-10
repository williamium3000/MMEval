#!/bin/bash
# Run DSG_v5_context_pope.py on specified conv scripts (2 LLM calls per conversation:
# objects from context+goal, then exist-match against scene graph).
# Output: same directory as input, with _pope_converted.json suffix.

set -e
export PYTHONPATH="${PYTHONPATH:-.}:./infer/LLaVA:./infer/LLaVA/llava"

eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate qwenvl 2>/dev/null || true

# Load .env so REMOTE_API_URL / REMOTE_API_KEY are set (same as DSG_v5_context_pope.py)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Check connection to endpoint (curl). Set CHECK_VERBOSE=1 to print the curl command.
# Uses REMOTE_API_MODEL from .env (same as DSG_v5_context_pope.py) so the server accepts the request.
check_endpoint() {
    if [ -z "${REMOTE_API_URL:-}" ]; then
        echo "REMOTE_API_URL not set. Set in .env or export before running."
        return 1
    fi
    # Build chat/completions URL: avoid double /v1/ when REMOTE_API_URL already ends with /v1
    API_URL="${REMOTE_API_URL%/}"
    if [[ "$API_URL" != *"/chat/completions"* ]]; then
        if [[ "$API_URL" == */v1 ]]; then
            API_URL="${API_URL}/chat/completions"
        else
            API_URL="${API_URL}/v1/chat/completions"
        fi
    fi
    MODEL="${REMOTE_API_MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
    echo "Checking connection to: $API_URL (model: $MODEL)"
    AUTH_HEADER=""
    [ -n "${REMOTE_API_KEY:-}" ] && AUTH_HEADER="Authorization: Bearer $REMOTE_API_KEY"
    PAYLOAD="{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}"
    if [ -n "${CHECK_VERBOSE:-}" ]; then
        echo "Curl command (run manually to test):"
        echo "  curl -s -w '\\nHTTP_CODE:%{http_code}' -X POST \"$API_URL\" \\"
        echo "    -H \"Content-Type: application/json\" \\"
        [ -n "$AUTH_HEADER" ] && echo "    -H \"Authorization: Bearer \$REMOTE_API_KEY\" \\"
        echo "    -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}' \\"
        echo "    --connect-timeout 10 --max-time 30"
    fi
    if [ -n "$AUTH_HEADER" ]; then
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $REMOTE_API_KEY" \
            -d "$PAYLOAD" \
            --connect-timeout 10 --max-time 30)
    else
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL" \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD" \
            --connect-timeout 10 --max-time 30)
    fi
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
        echo "Connection OK (HTTP $HTTP_CODE)"
        return 0
    else
        echo "Connection failed (HTTP $HTTP_CODE). Check REMOTE_API_URL, REMOTE_API_KEY, REMOTE_API_MODEL in .env"
        return 1
    fi
}

# Parse --check-only: only run endpoint check then exit (and show curl command)
for arg in "$@"; do
    if [ "$arg" = "--check-only" ] || [ "$arg" = "--check" ]; then
        CHECK_VERBOSE=1 check_endpoint || exit 1
        echo "Check passed. Run without --check-only to run conversion."
        exit 0
    fi
done

# Curl test endpoint before running conversion
if ! check_endpoint; then
    exit 1
fi

BASE="work_dirs/vg/final_run_v18_gpt4o_completed"
INPUTS=(
    # "$BASE/Qwen2.5-VL-7B-Instruct/Qwen2.5-VL-7B-Instruct.json"
    # "$BASE/InternVL2-8B/InternVL2-8B.json"
    # "$BASE/llava-1.5-7b-hf/llava-1.5-7b-hf.json"
    # "$BASE/opera-llava-1.5/opera-llava-1.5.json"
    "$BASE/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct.json"
)

run_one() {
    conv_script="$1"
    if [ ! -f "$conv_script" ]; then
        echo "[SKIP] Not found: $conv_script"
        return 0
    fi
    dir="$(dirname "$conv_script")"
    base="$(basename "$conv_script" .json)"
    outfile="$dir/${base}_pope_converted_v5.json"
    echo "[$(date +'%H:%M:%S')] Start: $conv_script -> $outfile"
    python examiner/DSG_v5_context_pope.py \
        --conv_script "$conv_script" \
        --outfile "$outfile" && echo "[$(date +'%H:%M:%S')] Done: $outfile" || echo "[$(date +'%H:%M:%S')] FAILED: $outfile"
}

set +e
for conv_script in "${INPUTS[@]}"; do
    run_one "$conv_script" &
done
wait
set -e
echo "All done."
