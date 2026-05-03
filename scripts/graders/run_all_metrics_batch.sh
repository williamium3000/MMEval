#!/usr/bin/env bash
#
# scripts/graders/run_all_metrics_batch.sh
#
# Loop run_all_metrics.sh over every model JSON under one or more directories
# (defaults: svg/ablation_baseline2 + svg/caption), then aggregate the
# all_metrics.json summaries into a single CSV at the parent directory.
#
# Usage:
#   bash scripts/graders/run_all_metrics_batch.sh                  # default dirs
#   bash scripts/graders/run_all_metrics_batch.sh DIR [DIR ...]
#
# Env knobs:
#   FIRST_N      (default 100)  passed to run_all_metrics.sh --first-n
#   SAMPLE_NUM   (default 100)  passed to run_all_metrics.sh --sample-num
#   METRICS      (default all)  passed to run_all_metrics.sh --metrics
#   FAIL_FAST    (default 0)    set 1 to abort on first per-file failure
#   PARALLEL     (default 1)    fan files out across N workers, each pinned to
#                               GPU index (i % NGPU). Per-file logs go to
#                               work_dirs/_grader_batch_logs/<base>.<ts>.log.
#   GPU_LIST     (default "0,1,2,3,4,5,6,7")  comma-separated GPU IDs to use

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

FIRST_N="${FIRST_N:-100}"
SAMPLE_NUM="${SAMPLE_NUM:-100}"
METRICS="${METRICS:-all}"
FAIL_FAST="${FAIL_FAST:-0}"
PARALLEL="${PARALLEL:-1}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

if [ "$#" -eq 0 ]; then
    set -- \
        "work_dirs/svg/ablation_baseline2" \
        "work_dirs/svg/caption"
fi

# Collect input files (top-level JSONs only, skipping known non-input variants).
files=()
for d in "$@"; do
    [ -d "$d" ] || { echo "warn: $d not a directory" >&2; continue; }
    while IFS= read -r f; do
        b=$(basename "$f")
        case "$b" in
            *_cache.json)            continue ;;
            hallucinated_words_*)    continue ;;
            *_pope_*)                continue ;;
            *_both_*)                continue ;;
            mmhal_*|chair_results*)  continue ;;
            *_evaluation.json)       continue ;;
            all_metrics*)            continue ;;
        esac
        files+=("$f")
    done < <(find "$d" -maxdepth 1 -type f -name "*.json" | sort)
done

n=${#files[@]}
if [ "$n" -eq 0 ]; then
    echo "no input files found in: $*" >&2
    exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NGPU=${#GPUS[@]}
[ "$PARALLEL" -gt "$NGPU" ] && PARALLEL=$NGPU

echo "============================================================"
echo " run_all_metrics_batch.sh"
echo "  dirs        : $*"
echo "  files found : $n"
echo "  first-n     : $FIRST_N"
echo "  sample-num  : $SAMPLE_NUM"
echo "  metrics     : $METRICS"
echo "  parallel    : $PARALLEL workers (gpu list: ${GPUS[*]})"
echo "============================================================"
for f in "${files[@]}"; do echo "  - $f"; done
echo ""

LOG_DIR="$ROOT_DIR/work_dirs/_grader_batch_logs"
mkdir -p "$LOG_DIR"
RUN_TS=$(date +%Y%m%d_%H%M%S)
BATCH_LOG="$LOG_DIR/batch_${RUN_TS}.log"
echo "Batch log: $BATCH_LOG"

run_one() {
    # $1 = file path  $2 = idx (1-based)  $3 = total  $4 = gpu index
    local f="$1" idx="$2" total="$3" gpu="$4"
    local base
    base=$(basename "$f" .json)
    local plog="$LOG_DIR/file_${base}.${RUN_TS}.log"
    {
        echo "============================================================"
        echo " [$idx/$total] $f   (gpu=$gpu)"
        echo "============================================================"
        local start_ts end_ts
        start_ts=$(date +%s)
        if CUDA_VISIBLE_DEVICES="$gpu" bash "$SCRIPT_DIR/run_all_metrics.sh" "$f" \
                --first-n "$FIRST_N" \
                --sample-num "$SAMPLE_NUM" \
                --metrics "$METRICS" 2>&1; then
            end_ts=$(date +%s)
            echo "[batch] $f OK in $((end_ts-start_ts))s"
            echo "OK"
        else
            end_ts=$(date +%s)
            echo "[batch] FAILED: $f after $((end_ts-start_ts))s"
            echo "FAIL"
        fi
    } > "$plog" 2>&1
    # Append to combined log atomically
    flock -w 30 "$BATCH_LOG.lock" -c "cat '$plog' >> '$BATCH_LOG'" || cat "$plog" >> "$BATCH_LOG"
    # Mirror per-file final status to stdout
    tail -n 5 "$plog" | grep -E "\[batch\]|OK|FAIL" | tail -n 2
}

ok=0; fail=0
if [ "$PARALLEL" -le 1 ]; then
    for i in "${!files[@]}"; do
        f="${files[$i]}"
        idx=$((i+1))
        gpu="${GPUS[0]}"
        run_one "$f" "$idx" "$n" "$gpu"
    done
else
    # Pinned slot-based scheduler: at most $PARALLEL jobs, each owning a GPU slot.
    # When a slot's job exits, dispatch the next file onto the same GPU.
    declare -a slot_pid slot_gpu
    for s in $(seq 0 $((PARALLEL-1))); do
        slot_pid[$s]=0
        slot_gpu[$s]="${GPUS[$((s % NGPU))]}"
    done
    queue_idx=0
    while [ $queue_idx -lt $n ] || ((${#slot_pid[@]} > 0)); do
        for s in $(seq 0 $((PARALLEL-1))); do
            pid=${slot_pid[$s]}
            if [ "$pid" -ne 0 ] && ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null
                slot_pid[$s]=0
            fi
            if [ "${slot_pid[$s]}" -eq 0 ] && [ "$queue_idx" -lt "$n" ]; then
                f="${files[$queue_idx]}"
                idx=$((queue_idx+1))
                gpu="${slot_gpu[$s]}"
                echo "[batch] dispatch [$idx/$n] gpu=$gpu  $f"
                ( run_one "$f" "$idx" "$n" "$gpu" ) &
                slot_pid[$s]=$!
                queue_idx=$((queue_idx+1))
            fi
        done
        # Exit condition: queue drained AND all slots idle
        all_idle=1
        for s in $(seq 0 $((PARALLEL-1))); do
            [ "${slot_pid[$s]}" -ne 0 ] && all_idle=0
        done
        if [ "$queue_idx" -ge "$n" ] && [ "$all_idle" -eq 1 ]; then break; fi
        sleep 5
    done
    # Tally OK/FAIL from per-file logs
    for f in "${files[@]}"; do
        base=$(basename "$f" .json)
        plog="$LOG_DIR/file_${base}.${RUN_TS}.log"
        if grep -q "^OK$" "$plog" 2>/dev/null; then ok=$((ok+1)); else fail=$((fail+1)); fi
    done
fi

echo ""
echo "============================================================"
echo " Aggregating summaries"
echo "============================================================"

# Gather every all_metrics.json under the requested dirs into a CSV per dir.
/usr/bin/python3 - "$LOG_DIR" "$RUN_TS" "$@" <<'PYEOF'
import csv, json, os, sys, glob
log_dir, ts, *dirs = sys.argv[1:]

cols = ['file','model','CHAIRi','Coverage_avg','mmhal_pct','GED_SG_Distance',
        'DELCON','SoftSPICE','HaELM_pct','Faith']

agg_csv = os.path.join(log_dir, f"all_metrics_summary_{ts}.csv")
with open(agg_csv, 'w', newline='') as fout:
    w = csv.DictWriter(fout, fieldnames=cols)
    w.writeheader()
    rows = []
    for d in dirs:
        for j in sorted(glob.glob(os.path.join(d, '*', 'all_metrics.json'))):
            try:
                m = json.load(open(j))
            except Exception as e:
                print(f"  warn: skip {j}: {e}")
                continue
            base = os.path.basename(os.path.dirname(j))
            row = {
                'file':            j,
                'model':           base,
                'CHAIRi':          m.get('CHAIRi'),
                'Coverage_avg':    m.get('Coverage_avg'),
                'mmhal_pct':       m.get('mmhal'),
                'GED_SG_Distance': m.get('GED_SG_Distance'),
                'DELCON':          m.get('DELCON'),
                'SoftSPICE':       m.get('SoftSPICE'),
                'HaELM_pct':       m.get('HaELM'),
                'Faith':           m.get('Faith'),
            }
            w.writerow(row)
            rows.append(row)
    print(f"  Aggregated CSV: {agg_csv} ({len(rows)} rows)")

    # Pretty print
    print("")
    print(f"{'model':40s} {'CHAIRi':>8s} {'Cov':>7s} {'mmhal%':>8s} {'GED':>8s} {'DELCON':>8s} {'SoftSp':>8s} {'HaELM%':>8s} {'Faith':>8s}")
    print('-'*120)
    def f(x): return '-' if x is None else (f'{x:.3f}' if isinstance(x,(int,float)) else str(x))
    for r in rows:
        print(f"{r['model']:40s} {f(r['CHAIRi']):>8s} {f(r['Coverage_avg']):>7s} {f(r['mmhal_pct']):>8s} "
              f"{f(r['GED_SG_Distance']):>8s} {f(r['DELCON']):>8s} {f(r['SoftSPICE']):>8s} {f(r['HaELM_pct']):>8s} {f(r['Faith']):>8s}")
PYEOF

echo ""
echo "Batch complete: ok=$ok fail=$fail (of $n)"
echo "Batch log: $BATCH_LOG"
