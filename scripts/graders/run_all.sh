For future grader runs (any one of):
  # All graders use local Qwen3 EXCEPT GED (uses uniapi gpt-4o-mini)
  export REMOTE_API_URL="http://localhost:8004/v1/chat/completions"
  export REMOTE_BASE_URL="http://localhost:8004/v1"
  export REMOTE_API_KEY="dummy"   # local vLLM ignores
  export REMOTE_API_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
  export GED_MODEL="gpt-4o-mini"
  export GED_API_BASE="https://api.uniapi.io/v1"
  export GED_API_KEY="$YOUR_UNIAPI_KEY"  # set externally; do not hardcode
  bash scripts/graders/run_all_metrics.sh <input.json>