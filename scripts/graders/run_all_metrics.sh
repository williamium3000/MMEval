#!/usr/bin/env bash
#
# scripts/graders/run_all_metrics.sh
#
# Unified runner that produces ALL eight metrics for one or more conversation JSONs:
#   CHAIRi (↑)         — grader/chair/chair_dyna_vg.py    -> hallucinated_words_<base>.json
#   Cov avg (↑)        — same as above (Coverage_avg)
#   mmhal (↑)          — grader/mmhal/mmhal_grader.py     -> mmhal_<base>.json
#   GED/SG Distance (↑)— grader/sg/graph_distance.py      -> <base>_sg_ged.json
#   DELCON (↑)         — grader/sg/graph_distance.py      -> <base>_sg_delta_con.json
#   SoftSp (↓)         — scripts/graders/llm_parser.py    -> <base>_sg_dir/spice_summary.json
#   HaELM (↑)          — grader/HaELM/haelm.py            -> <base>_haelm.json
#   Faith (↓)          — grader/faithscore/eval.py        -> <base>_faithscore/merged_all.json
#
# Outputs land next to each input JSON in <input_dir>/<base>/. Final summary at
# <input_dir>/<base>/all_metrics.json.
#
# Per-file parallelism: each file runs two tracks concurrently
#   TRACK_QWEN : CHAIR -> MMHAL -> HaELM -> Faith     (uses REMOTE_API_* / Qwen3)
#   TRACK_API  : SG_GED -> SG_DELCON -> SoftSPICE     (SG uses GED_API_*/uniapi gpt-4o-mini)
#                                                      SoftSPICE is local (no LLM)
#
# Multi-file parallelism: pass multiple inputs to fan out N file-workers concurrently.
#   PARALLEL env (default = #inputs, cap 8) controls the file-level concurrency.
#
# Usage:
#   bash scripts/graders/run_all_metrics.sh <input.json> [more.json …] \
#        [--first-n N] [--sample-num N] [--skip-existing] [--no-skip] [--metrics LIST]
#
#   PARALLEL=4 bash scripts/graders/run_all_metrics.sh dir1/*.json dir2/*.json
#
# Examples:
#   bash scripts/graders/run_all_metrics.sh \
#     work_dirs/vg/v19d5/gemma-3-12b-it_cache.json \
#     work_dirs/vg/v19d5/InternVL3-8B-Instruct_cache.json
#
# Env requirements (graders that lack their model checkpoints are auto-skipped
# and recorded as "unavailable" in the summary):
#   - REMOTE_API_URL / REMOTE_API_KEY / REMOTE_API_MODEL — Qwen3 endpoint for
#       chair / mmhal / haelm / faith (and SoftSPICE if it ever uses an LLM).
#   - GED_MODEL / GED_API_BASE / GED_API_KEY (optional) — separate endpoint for
#       SG_GED + SG_DELCON parsing (e.g. uniapi gpt-4o-mini); falls back to
#       REMOTE_API_* if unset.
#   - llava-v1.5-7b at data/checkpoints/llava-v1.5-7b (Faith — optional).
#   - llama-7b-hf + grader/HaELM/checkpoint (HaELM — optional).

set -u
set -o pipefail

# ── Resolve paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

# Load .env for REMOTE_API_URL / REMOTE_API_KEY, but DO NOT clobber pre-existing
# exports — caller can override.
if [ -f "$ROOT_DIR/.env" ]; then
    while IFS='=' read -r _k _v; do
        [[ "$_k" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
        _v="${_v%\"}"; _v="${_v#\"}"
        if [ -z "${!_k:-}" ]; then export "$_k=$_v"; fi
    done < <(grep -vE '^\s*(#|$)' "$ROOT_DIR/.env")
fi

# Pick conda envs
CHAIR_PY="/raid/miniconda3/envs/coneval-chair/bin/python"
MMHAL_PY="/raid/miniconda3/envs/coneval-easydetect/bin/python"
SG_PY="/raid/miniconda3/envs/coneval-easydetect2/bin/python"
HAELM_PY="/raid/icy/iris/.conda/envs/coneval-haelm/bin/python"
FAITH_PY="/raid/icy/iris/.conda/envs/coneval-faith/bin/python"
EXTRA_PKGS="$ROOT_DIR/.local_pkgs"

# ── Args ─────────────────────────────────────────────────────────────────────
INPUTS=()
FIRST_N=""
SAMPLE_NUM="100"
SKIP_EXISTING=true
METRICS="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --first-n)       FIRST_N="$2"; shift 2 ;;
        --sample-num)    SAMPLE_NUM="$2"; shift 2 ;;
        --metrics)       METRICS="$2"; shift 2 ;;
        --no-skip)       SKIP_EXISTING=false; shift ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        -h|--help)
            sed -n '1,40p' "$0"; exit 0 ;;
        *)
            INPUTS+=("$1"); shift ;;
    esac
done

if [ "${#INPUTS[@]}" -eq 0 ]; then
    echo "Error: no input JSONs given." >&2
    echo "Usage: $0 <input.json> [more.json …] [--first-n N] [--sample-num N]" >&2
    exit 1
fi

# Validate inputs
VALID_INPUTS=()
for f in "${INPUTS[@]}"; do
    if [ ! -f "$f" ]; then
        echo "warn: input not found, skipping: $f" >&2
        continue
    fi
    VALID_INPUTS+=("$(readlink -f "$f")")
done
if [ "${#VALID_INPUTS[@]}" -eq 0 ]; then
    echo "Error: none of the inputs exist." >&2; exit 1
fi

# Default sample slice for graders that support it
if [ -z "$FIRST_N" ]; then FIRST_N="$SAMPLE_NUM"; fi

# File-level parallelism
PARALLEL="${PARALLEL:-${#VALID_INPUTS[@]}}"
[ "$PARALLEL" -gt 8 ] && PARALLEL=8
[ "$PARALLEL" -gt "${#VALID_INPUTS[@]}" ] && PARALLEL=${#VALID_INPUTS[@]}

echo "============================================================"
echo " run_all_metrics.sh (parallel)"
echo "  inputs        : ${#VALID_INPUTS[@]} file(s)"
for f in "${VALID_INPUTS[@]}"; do echo "                  $f"; done
echo "  first-n       : $FIRST_N (mmhal/faith only — chair runs full file)"
echo "  sample-num    : $SAMPLE_NUM (sg/haelm)"
echo "  metrics       : $METRICS"
echo "  parallel      : $PARALLEL file-worker(s)"
echo "  GED endpoint  : ${GED_API_BASE:-<falls back to REMOTE_BASE_URL>} model=${GED_MODEL:-<same as REMOTE_API_MODEL>}"
echo "  Qwen endpoint : ${REMOTE_API_URL:-(unset!)} model=${REMOTE_API_MODEL:-Qwen3-30B-A3B-Instruct-2507}"
echo "============================================================"

# Helper: should we run a given metric?
want() {
    case "$METRICS" in
        all) return 0 ;;
        *) [[ ",$METRICS," == *",$1,"* ]] ;;
    esac
}

# Helper: file-exists+nonempty
nonempty() { [ -s "$1" ]; }

# ── Per-file processor ──────────────────────────────────────────────────────
process_one_file() {
    local INPUT_JSON="$1"
    local INPUT_DIR="$(dirname "$INPUT_JSON")"
    local BASE="$(basename "$INPUT_JSON" .json)"
    local OUT_DIR="$INPUT_DIR/$BASE"
    mkdir -p "$OUT_DIR"
    local TAG="[$BASE]"

    # Detect caption-style (no per-turn 'gt')
    local IS_CAPTION_STYLE=false
    if /usr/bin/python3 -c "
import json,sys
d=json.load(open('$INPUT_JSON'))
sample=d[0] if isinstance(d,list) and d else d
convs=sample.get('conversations') or []
has_gt=any('gt' in c for c in convs[:5])
sys.exit(0 if not has_gt else 1)
" 2>/dev/null; then
        IS_CAPTION_STYLE=true
    fi

    echo "$TAG ============================================================"
    echo "$TAG input         : $INPUT_JSON"
    echo "$TAG output dir    : $OUT_DIR"
    echo "$TAG caption-style : $IS_CAPTION_STYLE"

    # Output paths
    local CHAIR_OUT="$INPUT_DIR/hallucinated_words_${BASE}.json"
    local MMHAL_OUT="$OUT_DIR/mmhal_${BASE}.json"
    local MMHAL_OUT_SIBLING="$INPUT_DIR/mmhal_${BASE}.json"
    local SG_GED_OUT="$OUT_DIR/${BASE}_sg_ged.json"
    local SG_DC_OUT="$OUT_DIR/${BASE}_sg_delta_con.json"
    local SP_DIR="$OUT_DIR/softspice"
    local SP_SUMMARY="$SP_DIR/spice_summary.json"
    local HE_OUT="$OUT_DIR/${BASE}_haelm.json"
    local FS_DIR="$OUT_DIR/faithscore"
    local FS_OUT="$FS_DIR/merged_all.json"

    if [ ! -s "$MMHAL_OUT" ] && [ -s "$MMHAL_OUT_SIBLING" ]; then
        MMHAL_OUT="$MMHAL_OUT_SIBLING"
    fi

    # Per-step runners (defined as inner shells — `set -u` not propagated to bg jobs cleanly)
    local QWEN_LOG="$OUT_DIR/track_qwen.log"
    local API_LOG="$OUT_DIR/track_api.log"

    # ── TRACK_QWEN: CHAIR -> MMHAL -> HaELM -> Faith ──────────────────────────
    (
        exec >>"$QWEN_LOG" 2>&1
        echo "==== $TAG TRACK_QWEN start at $(date +%T) ===="

        # 1. CHAIR
        if want chair; then
            if [ "$SKIP_EXISTING" = true ] && nonempty "$CHAIR_OUT"; then
                echo "$TAG [CHAIR] reuse existing $CHAIR_OUT"
            else
                echo "$TAG [CHAIR] running chair_dyna_vg.py …"
                ( cd "$ROOT_DIR" && \
                  PYTHONPATH="$ROOT_DIR" \
                  "$CHAIR_PY" grader/chair/chair_dyna_vg.py \
                      "$INPUT_JSON" \
                      grader/chair/filtered_object_synsets_final.json \
                      --api-url "$REMOTE_API_URL" \
                      --api-key "$REMOTE_API_KEY" \
                      --model "${REMOTE_API_MODEL:-Qwen3-30B-A3B-Instruct-2507}" \
                      --group-by-image
                ) || echo "$TAG [CHAIR] failed (continuing)"
            fi
        fi

        # 2. MMHAL
        if want mmhal; then
            if [ "$SKIP_EXISTING" = true ] && nonempty "$MMHAL_OUT"; then
                echo "$TAG [MMHAL] reuse existing $MMHAL_OUT"
            else
                echo "$TAG [MMHAL] running mmhal_grader.py … (workers=${MMHAL_WORKERS:-16})"
                local gt_type_arg=""
                [ "$IS_CAPTION_STYLE" = true ] && gt_type_arg="--gt-type caption"
                local first_n_arg=""
                [ -n "$FIRST_N" ] && first_n_arg="--first-n $FIRST_N"
                local no_skip_arg=""
                [ "$SKIP_EXISTING" = false ] && no_skip_arg="--no-skip"
                ( cd "$ROOT_DIR" && \
                  PYTHONPATH="$ROOT_DIR" \
                  "$MMHAL_PY" grader/mmhal/mmhal_grader.py \
                      --response   "$INPUT_JSON" \
                      --evaluation "$MMHAL_OUT" \
                      --gpt-model  "${MMHAL_MODEL:-${REMOTE_API_MODEL:-Llama-3.1-70B-Instruct}}" \
                      --api-url    "$REMOTE_API_URL" \
                      --api-key    "$REMOTE_API_KEY" \
                      --workers    "${MMHAL_WORKERS:-16}" \
                      $no_skip_arg $gt_type_arg $first_n_arg
                ) || echo "$TAG [MMHAL] failed (continuing)"
            fi
        fi

        # 3. HaELM (local GPU)
        if want haelm; then
            if [ "$SKIP_EXISTING" = true ] && nonempty "$HE_OUT"; then
                echo "$TAG [HaELM] reuse existing $HE_OUT"
            else
                local LLAMA_PATH="${LLAMA_PATH:-data/checkpoints/llama-7b-hf}"
                local HE_CKPT="${HE_CKPT:-data/checkpoints/llama-7b}"
                if [ ! -d "$LLAMA_PATH" ] || [ ! -d "$HE_CKPT" ]; then
                    echo "$TAG [HaELM] checkpoints missing — skipping."
                else
                    echo "$TAG [HaELM] running haelm.py … (GPU=${HAELM_GPU:-2})"
                    ( cd "$ROOT_DIR" && \
                      PYTHONNOUSERSITE=1 \
                      PYTHONPATH="$ROOT_DIR" \
                      CUDA_VISIBLE_DEVICES="${HAELM_GPU:-2}" \
                      "$HAELM_PY" grader/HaELM/haelm.py \
                          --conv "$INPUT_JSON" \
                          --llama_path "$LLAMA_PATH" \
                          --checkpoint_path "$HE_CKPT" \
                          --outfile "$HE_OUT" \
                          --sample_num "$SAMPLE_NUM" 2>&1 | tee "$OUT_DIR/haelm.log"
                    ) || echo "$TAG [HaELM] failed (continuing)"
                fi
            fi
        fi

        # 4. Faith — DISABLED (kept for reference; Faith stage-1 hangs and is no longer run)
        # if want faith; then
        #     if [ "$SKIP_EXISTING" = true ] && nonempty "$FS_OUT"; then
        #         echo "$TAG [Faith] reuse existing $FS_OUT"
        #     else
        #         local LLAVA_PATH="${LLAVA_PATH:-data/checkpoints/llava-v1.5-7b}"
        #         if ! "$FAITH_PY" -c "import llava" 2>/dev/null; then
        #             echo "$TAG [Faith] llava not installed in $FAITH_PY — skipping."
        #         elif [ ! -d "$LLAVA_PATH" ]; then
        #             echo "$TAG [Faith] LLaVA checkpoint missing — skipping."
        #         else
        #             echo "$TAG [Faith] running faithscore/eval.py …"
        #             mkdir -p "$FS_DIR"
        #             ( cd "$ROOT_DIR/grader/faithscore" && \
        #               PYTHONNOUSERSITE=1 \
        #               PYTHONPATH="$ROOT_DIR/grader/faithscore:$ROOT_DIR" \
        #               OPENAI_API_KEY="$REMOTE_API_KEY" \
        #               OPENAI_BASE_URL="${REMOTE_BASE_URL:-${REMOTE_API_URL%/chat/completions}}" \
        #               "$FAITH_PY" eval.py --mode eval \
        #                   --conv "$INPUT_JSON" \
        #                   --vem_type llava --llava_path "$ROOT_DIR/$LLAVA_PATH" \
        #                   --openai_model "${REMOTE_API_MODEL:-Qwen3-30B-A3B-Instruct-2507}" \
        #                   --sample_num "$FIRST_N" \
        #                   --save_judgments "$FS_OUT" 2>&1 | tee "$OUT_DIR/faithscore.log"
        #             ) || echo "$TAG [Faith] failed (continuing)"
        #         fi
        #     fi
        # fi
        echo "==== $TAG TRACK_QWEN end at $(date +%T) ===="
    ) &
    local pid_qwen=$!

    # ── TRACK_API: SG_GED -> SG_DELCON -> SoftSPICE ──────────────────────────
    (
        exec >>"$API_LOG" 2>&1
        echo "==== $TAG TRACK_API start at $(date +%T) ===="

        # SG runner closure
        run_sg() {
            local method="$1" outfile="$2"
            if [ "$SKIP_EXISTING" = true ] && nonempty "$outfile" && \
               /usr/bin/python3 -c "
import json,sys
try:
    d=json.load(open('$outfile'))
    n=sum(1 for x in d if isinstance(x,dict) and x.get('dist_score') is not None)
    sys.exit(0 if n>0 else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
                echo "$TAG [SG/$method] reuse existing $outfile"
                return 0
            fi
            echo "$TAG [SG/$method] running graph_distance.py …"
            local _llm_model="${REMOTE_API_MODEL:-Qwen3-30B-A3B-Instruct-2507}"
            local _api_key="$REMOTE_API_KEY"
            local _api_base="${REMOTE_BASE_URL:-${REMOTE_API_URL%/chat/completions}}"
            if [ -n "${GED_MODEL:-}" ]; then
                _llm_model="$GED_MODEL"
                [ -n "${GED_API_BASE:-}" ] && _api_base="$GED_API_BASE"
                [ -n "${GED_API_KEY:-}"  ] && _api_key="$GED_API_KEY"
            fi
            ( cd "$ROOT_DIR" && \
              PYTHONNOUSERSITE=1 \
              PYTHONPATH="$EXTRA_PKGS:$ROOT_DIR" \
              OPENAI_API_KEY="$_api_key" \
              OPENAI_BASE_URL="$_api_base" \
              "$SG_PY" grader/sg/graph_distance.py \
                  --conv_script "$INPUT_JSON" \
                  --outdir "$OUT_DIR" \
                  --output_file "$(basename "$outfile")" \
                  --method "$method" \
                  --sample_num "$SAMPLE_NUM" \
                  --max_workers 8 \
                  --distance_workers 8 \
                  --llm_model "$_llm_model" \
                  --merge_first
            ) || echo "$TAG [SG/$method] failed (continuing)"
        }

        # 1. SG_GED
        if want sg_ged; then run_sg ged "$SG_GED_OUT"; fi

        # 2. SG_DELCON (seed cache from ged, strip dist_score)
        if want sg_delcon; then
            if [ -s "$SG_GED_OUT" ] && [ ! -s "$SG_DC_OUT" ]; then
                /usr/bin/python3 -c "
import json
src,dst='$SG_GED_OUT','$SG_DC_OUT'
d=json.load(open(src))
for s in d:
    if isinstance(s,dict) and 'dist_score' in s:
        del s['dist_score']
json.dump(d, open(dst,'w'), indent=4)
"
            fi
            run_sg delta_con "$SG_DC_OUT"
        fi

        # 3. SoftSPICE — DISABLED (no longer reported; metric correlation low)
        # if want softsp; then
        #     if [ "$SKIP_EXISTING" = true ] && nonempty "$SP_SUMMARY"; then
        #         echo "$TAG [SoftSPICE] reuse existing $SP_SUMMARY"
        #     else
        #         echo "$TAG [SoftSPICE] running llm_parser.py --sg_dir …"
        #         mkdir -p "$SP_DIR"
        #         for f in "$SG_GED_OUT" "$SG_DC_OUT"; do
        #             [ -f "$f" ] && cp -f "$f" "$SP_DIR/$(basename "$f")"
        #         done
        #         if ls "$SP_DIR"/*.json >/dev/null 2>&1; then
        #             ( cd "$ROOT_DIR" && \
        #               PYTHONNOUSERSITE=1 \
        #               PYTHONPATH="$EXTRA_PKGS:$ROOT_DIR" \
        #               "$SG_PY" scripts/graders/llm_parser.py \
        #                   --sg_dir "$SP_DIR" \
        #                   --metric all
        #             ) || echo "$TAG [SoftSPICE] failed (continuing)"
        #         else
        #             echo "$TAG [SoftSPICE] no SG outputs to score (skipping)"
        #         fi
        #     fi
        # fi
        echo "==== $TAG TRACK_API end at $(date +%T) ===="
    ) &
    local pid_api=$!

    # Wait for both tracks
    local rc=0
    wait "$pid_qwen" || rc=$?
    wait "$pid_api"  || rc=$?

    # ── Summary ────────────────────────────────────────────────────────────
    local SUMMARY_JSON="$OUT_DIR/all_metrics.json"
    local SUMMARY_TXT="$OUT_DIR/all_metrics.txt"

    /usr/bin/python3 - "$INPUT_JSON" "$OUT_DIR" "$CHAIR_OUT" "$MMHAL_OUT" "$SG_GED_OUT" "$SG_DC_OUT" "$SP_SUMMARY" "$HE_OUT" "$FS_OUT" "$SUMMARY_JSON" "$SUMMARY_TXT" <<'PYEOF'
import json, os, re, sys

(_, in_path, out_dir, chair_p, mmhal_p, ged_p, dc_p, sp_p, he_p, fs_p, sj, st) = sys.argv

def load(p):
    try:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return json.load(open(p, 'r', encoding='utf-8'))
    except Exception as e:
        print(f"  warn: failed to load {p}: {e}")
    return None

def fmt(v, n=4):
    return ("-" if v is None else (f"{v:.{n}f}" if isinstance(v, (int, float)) else str(v)))

def avg_dist(d):
    if not d: return None
    if isinstance(d, list):
        scores = [x.get('dist_score') for x in d if isinstance(x, dict) and x.get('dist_score') is not None]
        return (sum(scores) / len(scores)) if scores else None
    return None

chair = load(chair_p)
chair_i = (chair or {}).get('overall_metrics', {}).get('CHAIRi_v2')
cov_avg = (chair or {}).get('overall_metrics', {}).get('Coverage_avg')

mmhal = load(mmhal_p)
_avg = (mmhal or {}).get('overall_metrics', {}).get('avg_score')
# mmhal as HALLUCINATION RATE (lower-is-better): (6 - avg_score) / 6 * 100
# avg_score is on 0-6 scale where 6 = perfect (no hallucination).
mmhal_score = ((6.0 - _avg) / 6.0 * 100.0) if isinstance(_avg, (int, float)) else None

ged_avg = avg_dist(load(ged_p))
dc_avg = avg_dist(load(dc_p))

sp = load(sp_p)
soft_sp = None
if sp and isinstance(sp, dict):
    res = sp.get('results') or []
    ws = [r for r in res if r.get('mean_soft_spice') is not None and r.get('n')]
    if ws:
        tot = sum(r['n'] for r in ws)
        soft_sp = sum(r['mean_soft_spice']*r['n'] for r in ws) / tot

he = load(he_p)
he_rate = None
log = os.path.join(out_dir, 'haelm.log')
if os.path.exists(log):
    txt = open(log, 'r', encoding='utf-8', errors='ignore').read()
    m = re.search(r'Sentence level hallucination rate:\s*([\d.]+)', txt)
    # HaELM as raw sentence hallucination rate %, examiner perspective (higher = better):
    # higher rate = more sentence-level hallucinations exposed = better examiner.
    if m: he_rate = float(m.group(1)) * 100.0

fs = load(fs_p)
faith = None
if fs and isinstance(fs, dict):
    faith = fs.get('faithscore') or fs.get('overall_score')
log = os.path.join(out_dir, 'faithscore.log')
if faith is None and os.path.exists(log):
    txt = open(log, 'r', encoding='utf-8', errors='ignore').read()
    m = re.search(r'Overall score:\s*([\d.]+)', txt)
    if m: faith = float(m.group(1))

summary = {
    'input':           in_path,
    'CHAIRi':          chair_i,
    'Coverage_avg':    cov_avg,
    'mmhal':           mmhal_score,
    'GED_SG_Distance': ged_avg,
    'DELCON':          dc_avg,
    'SoftSPICE':       soft_sp,
    'HaELM':           he_rate,
    'Faith':           faith,
}
json.dump(summary, open(sj, 'w'), indent=2)

rows = [
    ('CHAIRi (↑)',                       summary['CHAIRi']),
    ('Cov avg (↑)',                      summary['Coverage_avg']),
    ('mmhal hallucination % (↑)',        summary['mmhal']),    # examiner perspective: more = better
    ('GED/SG Distance (↑)',              summary['GED_SG_Distance']),
    ('DELCON (↑)',                       summary['DELCON']),
    ('SoftSp (↓)',                       summary['SoftSPICE']),
    ('HaELM hallucination % (↑)',        summary['HaELM']),    # raw sentence_hall_rate, higher = better examiner
    ('Faith (↓)',                        summary['Faith']),
]
lines = ['=== ALL METRICS ===', f'input: {in_path}', '']
for name, val in rows:
    lines.append(f'  {name:24s} = {fmt(val)}')
out = '\n'.join(lines) + '\n'
open(st, 'w').write(out)
print('\n' + out)
print(f'  Summary JSON: {sj}')
print(f'  Summary TXT : {st}')
PYEOF
    echo "$TAG done -> $SUMMARY_JSON"
    return $rc
}

# ── File-level dispatcher ───────────────────────────────────────────────────
# Run up to PARALLEL files concurrently. Each file forks 2 internal tracks.
declare -a PIDS=()
declare -A PID2BASE=()
running=0
i=0

reap_one() {
    local pid="$1"
    local base="${PID2BASE[$pid]:-?}"
    if wait "$pid"; then
        echo "[main] [$base] file done OK"
    else
        echo "[main] [$base] file FAILED (rc=$?)"
    fi
    unset PID2BASE[$pid]
}

while [ "$i" -lt "${#VALID_INPUTS[@]}" ] || [ "$running" -gt 0 ]; do
    while [ "$running" -lt "$PARALLEL" ] && [ "$i" -lt "${#VALID_INPUTS[@]}" ]; do
        f="${VALID_INPUTS[$i]}"
        b="$(basename "$f" .json)"
        echo "[main] launch worker for $b ($((i+1))/${#VALID_INPUTS[@]})"
        process_one_file "$f" &
        pid=$!
        PIDS+=("$pid")
        PID2BASE[$pid]="$b"
        running=$((running+1))
        i=$((i+1))
    done
    if [ "$running" -gt 0 ]; then
        # wait for any one to finish
        wait -n
        # rebuild PIDS keeping only still-alive
        new_pids=()
        for p in "${PIDS[@]}"; do
            if kill -0 "$p" 2>/dev/null; then
                new_pids+=("$p")
            fi
        done
        running=${#new_pids[@]}
        PIDS=("${new_pids[@]}")
    fi
done

echo "============================================================"
echo " run_all_metrics.sh: all ${#VALID_INPUTS[@]} file(s) done"
echo "============================================================"
