#!/usr/bin/env bash
# scripts/run_graders.sh
# Unified wrapper to run SG Distance, FaithScore, HaELM, and/or SoftSPICE graders.
#
# Usage (run from MMEval project root):
#   ./scripts/run_graders.sh --input_dir <dir> [options]
#
# Toggle (if none of --sg / --faithscore / --haelm is given, those THREE are run;
# --spice is always opt-in because it needs the factual_scene_graph deps):
#   --sg                Enable SG Distance grader
#   --faithscore        Enable FaithScore grader
#   --haelm             Enable HaELM grader
#   --spice             Enable SoftSPICE grader (SPICE + Soft-SPICE summary)
#
# Common options:
#   --input_dir DIR     Directory of JSON files.
#                       SG        : processes every *.json in this dir.
#                       FaithScore: iterates over *.json unless --conv is given.
#                       HaELM     : iterates over *.json unless --conv is given.
#   --conv FILE         Single JSON file for FaithScore / HaELM (overrides --input_dir).
#   --output_dir DIR    Base output directory (default: output/grader_results).
#   -j N, --jobs N      File-level parallelism shared by SG and HaELM (default: 1).
#                       FaithScore has its own --fs_jobs for batch-level parallelism.
#
# SG-specific options (forwarded to graders/sg/graph_distance.py):
#   --sg_outdir DIR       SG output dir (default: <output_dir>/sg)
#   --sg_method METHOD    Distance method: delta_con|ged|netlsd|portrait|hamming|
#                         him|frobenius|graph_diffusion|jaccard|netsimile|resistance
#                         (default: delta_con)
#   --sg_sample_num N     Max samples per file (default: 100)
#   --sg_timeout N        GED timeout in seconds (default: 300)
#   --merge_first         Merge records by image_id before computing distances
#   --distance_only       Skip LLM parsing; input files must already have unique_sg
#   --sg_workers N        Parallel threads inside graph_distance.py (default: 10)
#   --sg_include GLOB     Only process SG files whose basename matches this glob
#   --sg_exclude GLOB     Skip SG files whose basename matches this glob
#   --sg_flush N          Flush intermediate results every N samples (default: 0 = only at end)
#
# FaithScore-specific options (forwarded to graders/faithscore/eval.py):
#   --fs_outdir DIR       FaithScore output dir (default: <output_dir>/faithscore)
#   --fs_num_jobs N       Number of index-range batches to split each conv file into (default: 4)
#   --fs_jobs N           Local FaithScore batch parallelism (default: fs_num_jobs)
#   --vem_type TYPE       Visual entailment model: llava|ofa|ofa-ve (default: llava)
#   --llava_path PATH     Path to LLaVA checkpoint directory (required for --vem_type llava)
#   --openai_model MODEL  OpenAI model for stages 1+2 (default: gpt-4o)
#   --openai_base_url URL OpenAI-compatible base URL (default: https://api.openai.com/v1)
#   --fs_merge            Merge per-batch FaithScore results into merged_all.json when done
#
# HaELM-specific options (forwarded to graders/HaELM/haelm.py):
#   --haelm_outdir DIR      HaELM output dir (default: <output_dir>/haelm)
#   --haelm_llama_path PATH LLaMA base model path (default: checkpoints/llama-7b-hf)
#   --haelm_checkpoint PATH HaELM LoRA checkpoint path (default: graders/HaELM/checkpoint)
#   --haelm_sample_num N    Max samples per file (default: 100)
#
# SoftSPICE-specific options (forwarded to graders/sg/llm_parser.py --sg_dir):
#   Inputs must already have unique_sg from the SG grader (LLM-parsed). This
#   wrapper does NOT expose any factual_scene_graph parser knobs — parsing is
#   the SG grader's job.
#   --spice_dir DIR         Directory of JSON files with unique_sg already populated.
#                           Default: <sg_outdir> if --sg also ran, else <input_dir>.
#                           Summary is written to <spice_dir>/spice_summary.{json,txt}.
#   --spice_text_encoder M  Text encoder used for Soft-SPICE embeddings (default: all-MiniLM-L6-v2)
#   --spice_metric M        all|set_match|spice|soft_spice (default: all)
#
# Examples:
#   # Run ALL three default graders (SG + FaithScore + HaELM) on a directory
#   # (SoftSPICE is opt-in and not included in the no-toggle default)
#   ./scripts/run_graders.sh \
#     --input_dir output/context \
#     --llava_path checkpoints/llava-v1.5-7b \
#     --haelm_llama_path checkpoints/llama-7b-hf \
#     --haelm_checkpoint graders/HaELM/checkpoint
#
#   # SG only, 4 parallel files, custom method
#   ./scripts/run_graders.sh --sg \
#     --input_dir output/context \
#     --sg_method ged -j 4
#
#   # FaithScore only on a specific file, 4 batches, 2 in parallel, then merge
#   ./scripts/run_graders.sh --faithscore \
#     --conv output/context/InternVL2-8B.json \
#     --fs_num_jobs 4 --fs_jobs 2 --fs_merge \
#     --llava_path checkpoints/llava-v1.5-7b
#
#   # HaELM only on a directory
#   ./scripts/run_graders.sh --haelm \
#     --input_dir output/context \
#     --haelm_llama_path checkpoints/llama-7b-hf \
#     --haelm_checkpoint graders/HaELM/checkpoint \
#     --haelm_sample_num 50
#
#   # SG + HaELM together, skip FaithScore
#   ./scripts/run_graders.sh --sg --haelm \
#     --input_dir output/context \
#     --haelm_llama_path checkpoints/llama-7b-hf \
#     --haelm_checkpoint graders/HaELM/checkpoint
#
#   # SG → SoftSPICE in one shot. SG writes unique_sg to <output_dir>/sg/, then
#   # SoftSPICE auto-points --spice_dir there.
#   ./scripts/run_graders.sh --sg --spice \
#     --input_dir output/context
#
#   # SoftSPICE only, on a dir whose JSON files already contain unique_sg
#   # from a prior SG run. Writes spice_summary.{json,txt} into that dir.
#   ./scripts/run_graders.sh --spice \
#     --spice_dir output/grader_results/sg
#
#   # SoftSPICE only, soft_spice metric, custom sentence-transformers encoder
#   ./scripts/run_graders.sh --spice \
#     --spice_dir output/grader_results/sg \
#     --spice_metric soft_spice \
#     --spice_text_encoder all-mpnet-base-v2

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
PYTHON="${PYTHON:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo 'python3')}"

SG_SCRIPT="${ROOT_DIR}/graders/sg/graph_distance.py"
FS_SCRIPT="${ROOT_DIR}/graders/faithscore/eval.py"
HE_SCRIPT="${ROOT_DIR}/graders/HaELM/haelm.py"
SP_SCRIPT="${ROOT_DIR}/graders/sg/llm_parser.py"

# ── Toggle flags ──────────────────────────────────────────────────────────────
RUN_SG=false
RUN_FS=false
RUN_HE=false
RUN_SP=false

# ── Common ────────────────────────────────────────────────────────────────────
INPUT_DIR=""
CONV_FILE=""
OUTPUT_DIR="output/grader_results"
JOBS=1

# ── SG args ───────────────────────────────────────────────────────────────────
SG_OUTDIR=""
SG_METHOD="delta_con"
SG_SAMPLE_NUM=100
SG_TIMEOUT=300
SG_MERGE_FIRST=false
SG_DISTANCE_ONLY=false
SG_WORKERS=10
SG_INCLUDE_GLOB=""
SG_EXCLUDE_GLOB=""
SG_FLUSH=0

# ── FaithScore args ───────────────────────────────────────────────────────────
FS_OUTDIR=""
FS_NUM_JOBS=4
FS_JOBS=""          # defaults to FS_NUM_JOBS after parsing
FS_VEM_TYPE="llava"
FS_LLAVA_PATH="checkpoints/llava-v1.5-7b"
FS_OPENAI_MODEL="gpt-4o"
FS_OPENAI_BASE_URL="https://api.openai.com/v1"
FS_MERGE=false

# ── HaELM args ────────────────────────────────────────────────────────────────
HE_OUTDIR=""
HE_LLAMA_PATH="checkpoints/llama-7b-hf"
HE_CHECKPOINT="graders/HaELM/checkpoint"
HE_SAMPLE_NUM=100

# ── SoftSPICE args ────────────────────────────────────────────────────────────
SP_INDIR=""
SP_TEXT_ENCODER="all-MiniLM-L6-v2"
SP_METRIC="all"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
  case "$1" in
    # Toggle
    --sg)                RUN_SG=true;  shift ;;
    --faithscore)        RUN_FS=true;  shift ;;
    --haelm)             RUN_HE=true;  shift ;;
    --spice)             RUN_SP=true;  shift ;;
    # Common
    --input_dir)         INPUT_DIR="$2";          shift 2 ;;
    --conv)              CONV_FILE="$2";           shift 2 ;;
    --output_dir)        OUTPUT_DIR="$2";          shift 2 ;;
    -j)                  JOBS="${2:-4}";           shift 2 ;;
    --jobs)              JOBS="${2:-4}";           shift 2 ;;
    -j[0-9]*)            JOBS="${1#-j}";           shift ;;
    # SG-specific
    --sg_outdir)         SG_OUTDIR="$2";           shift 2 ;;
    --sg_method)         SG_METHOD="$2";           shift 2 ;;
    --sg_sample_num)     SG_SAMPLE_NUM="$2";       shift 2 ;;
    --sg_timeout)        SG_TIMEOUT="$2";          shift 2 ;;
    --merge_first)       SG_MERGE_FIRST=true;      shift ;;
    --distance_only)     SG_DISTANCE_ONLY=true;    shift ;;
    --sg_workers)        SG_WORKERS="$2";          shift 2 ;;
    --sg_include)        SG_INCLUDE_GLOB="$2";     shift 2 ;;
    --sg_exclude)        SG_EXCLUDE_GLOB="$2";     shift 2 ;;
    --sg_flush)          SG_FLUSH="$2";            shift 2 ;;
    # FaithScore-specific
    --fs_outdir)         FS_OUTDIR="$2";           shift 2 ;;
    --fs_num_jobs)       FS_NUM_JOBS="$2";         shift 2 ;;
    --fs_jobs)           FS_JOBS="$2";             shift 2 ;;
    --vem_type)          FS_VEM_TYPE="$2";         shift 2 ;;
    --llava_path)        FS_LLAVA_PATH="$2";       shift 2 ;;
    --openai_model)      FS_OPENAI_MODEL="$2";     shift 2 ;;
    --openai_base_url)   FS_OPENAI_BASE_URL="$2";  shift 2 ;;
    --fs_merge)          FS_MERGE=true;            shift ;;
    # HaELM-specific
    --haelm_outdir)      HE_OUTDIR="$2";           shift 2 ;;
    --haelm_llama_path)  HE_LLAMA_PATH="$2";       shift 2 ;;
    --haelm_checkpoint)  HE_CHECKPOINT="$2";       shift 2 ;;
    --haelm_sample_num)  HE_SAMPLE_NUM="$2";       shift 2 ;;
    # SoftSPICE-specific
    --spice_dir)         SP_INDIR="$2";            shift 2 ;;
    --spice_text_encoder) SP_TEXT_ENCODER="$2";    shift 2 ;;
    --spice_metric)      SP_METRIC="$2";           shift 2 ;;
    *)
      echo "Unknown argument: $1"
      echo "Run with no args to see usage."
      exit 1
      ;;
  esac
done

# If no toggle was set, enable SG/FS/HE (SoftSPICE stays opt-in).
if [ "$RUN_SG" = false ] && [ "$RUN_FS" = false ] && [ "$RUN_HE" = false ] && [ "$RUN_SP" = false ]; then
  RUN_SG=true
  RUN_FS=true
  RUN_HE=true
fi

# Resolve default sub-output dirs
[ -z "$SG_OUTDIR" ] && SG_OUTDIR="${OUTPUT_DIR}/sg"
[ -z "$FS_OUTDIR" ] && FS_OUTDIR="${OUTPUT_DIR}/faithscore"
[ -z "$HE_OUTDIR" ] && HE_OUTDIR="${OUTPUT_DIR}/haelm"
[ -z "$FS_JOBS"   ] && FS_JOBS=$FS_NUM_JOBS

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo " run_graders.sh"
echo "  SG Distance : $RUN_SG"
echo "  FaithScore  : $RUN_FS"
echo "  HaELM       : $RUN_HE"
echo "  SoftSPICE   : $RUN_SP"
echo "  input_dir   : ${INPUT_DIR:-<not set>}"
echo "  output_dir  : $OUTPUT_DIR"
echo "  parallelism : -j $JOBS"
echo "============================================================"


# ═════════════════════════════════════════════════════════════════
#  SG DISTANCE
# ═════════════════════════════════════════════════════════════════
if [ "$RUN_SG" = true ]; then
  if [ -z "$INPUT_DIR" ] || [ ! -d "$INPUT_DIR" ]; then
    echo "[SG] Error: --input_dir must be a valid directory."
    exit 1
  fi

  mkdir -p "$SG_OUTDIR"
  echo ""
  echo "──────────────────────────────────────────────────────────"
  echo " SG Distance"
  echo "  input_dir  : $INPUT_DIR"
  echo "  outdir     : $SG_OUTDIR"
  echo "  method     : $SG_METHOD"
  echo "  sample_num : $SG_SAMPLE_NUM"
  echo "  jobs       : $JOBS"
  echo "──────────────────────────────────────────────────────────"

  SG_EXTRA=()
  SG_EXTRA+=(--method "$SG_METHOD")
  SG_EXTRA+=(--sample_num "$SG_SAMPLE_NUM")
  SG_EXTRA+=(--timeout "$SG_TIMEOUT")
  SG_EXTRA+=(--max_workers "$SG_WORKERS")
  SG_EXTRA+=(--distance_workers "$SG_WORKERS")
  [ "$SG_MERGE_FIRST"   = true ] && SG_EXTRA+=(--merge_first)
  [ "$SG_DISTANCE_ONLY" = true ] && SG_EXTRA+=(--distance_only)
  [ "$SG_FLUSH" -gt 0          ] && SG_EXTRA+=(--flush_interval "$SG_FLUSH")

  sg_should_process() {
    local base
    base=$(basename "$1")
    if [ -n "$SG_INCLUDE_GLOB" ]; then
      [[ $base == $SG_INCLUDE_GLOB ]] || return 1
    fi
    if [ -n "$SG_EXCLUDE_GLOB" ]; then
      [[ $base == $SG_EXCLUDE_GLOB ]] && return 1
    fi
    return 0
  }

  sg_run_one() {
    local f="$1"
    local base
    base=$(basename "$f" .json)
    local outname="${base}_sg.json"
    echo "  [SG] $f -> $SG_OUTDIR/$outname"
    "$PYTHON" "$SG_SCRIPT" \
      --conv_script "$f" \
      --outdir "$SG_OUTDIR" \
      --output_file "$outname" \
      "${SG_EXTRA[@]}"
    # Write per-file summary fragment
    "$PYTHON" - "$SG_OUTDIR/$outname" "$f" "$SG_OUTDIR/.summary_${base}.json" <<'PYEOF'
import json, sys, os
outpath, inpath, sumpath = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(outpath, encoding='utf-8'))
scores = [x.get('dist_score') for x in d if x.get('dist_score') is not None]
n = len(scores)
avg = round(sum(scores)/n, 4) if n else None
rec = {'input_file': os.path.basename(inpath), 'output_file': os.path.basename(outpath),
       'num_samples': n, 'avg_distance': avg}
with open(sumpath, 'w', encoding='utf-8') as out:
    out.write(json.dumps(rec, ensure_ascii=False))
PYEOF
  }

  sg_count=0
  if [ "$JOBS" -le 1 ]; then
    for f in "$INPUT_DIR"/*.json; do
      [ -f "$f" ] || continue
      sg_should_process "$f" || continue
      sg_run_one "$f"
      ((sg_count++)) || true
    done
  else
    for f in "$INPUT_DIR"/*.json; do
      [ -f "$f" ] || continue
      sg_should_process "$f" || continue
      while [ "$(jobs -r 2>/dev/null | wc -l)" -ge "$JOBS" ]; do sleep 0.3; done
      ( sg_run_one "$f" ) &
      ((sg_count++)) || true
    done
    _sg_fail=0
    for _job in $(jobs -p); do wait "$_job" || _sg_fail=$?; done
    [ "$_sg_fail" -ne 0 ] && echo "[SG] Warning: one or more parallel jobs failed (last exit: $_sg_fail)"
  fi

  # Merge summary fragments
  if [ "$sg_count" -gt 0 ]; then
    summary_count=$(find "$SG_OUTDIR" -maxdepth 1 -name '.summary_*.json' 2>/dev/null | wc -l || true)
    if [ "$summary_count" -gt 0 ]; then
      "$PYTHON" - "$SG_OUTDIR" <<'PYEOF'
import json, sys, os, glob
outdir = sys.argv[1]
recs = []
for p in sorted(glob.glob(os.path.join(outdir, '.summary_*.json'))):
    try:
        recs.append(json.loads(open(p, encoding='utf-8').read()))
    except Exception:
        pass
with open(os.path.join(outdir, 'sg_distance_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(recs, f, ensure_ascii=False, indent=2)
with open(os.path.join(outdir, 'sg_distance_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('input_file\toutput_file\tnum_samples\tavg_distance\n')
    for r in recs:
        f.write('{}\t{}\t{}\t{}\n'.format(
            r['input_file'], r['output_file'],
            r.get('num_samples',''), r.get('avg_distance','')))
for p in glob.glob(os.path.join(outdir, '.summary_*.json')):
    try: os.remove(p)
    except Exception: pass
PYEOF
      echo ""
      echo "[SG] Summary: $SG_OUTDIR/sg_distance_summary.json"
      cat "$SG_OUTDIR/sg_distance_summary.txt"
    fi
  fi
  echo "[SG] Done: $sg_count file(s) processed -> $SG_OUTDIR"
fi


# ═════════════════════════════════════════════════════════════════
#  FAITHSCORE
# ═════════════════════════════════════════════════════════════════
if [ "$RUN_FS" = true ]; then
  echo ""
  echo "──────────────────────────────────────────────────────────"
  echo " FaithScore"
  echo "  vem_type   : $FS_VEM_TYPE"
  echo "  num_jobs   : $FS_NUM_JOBS"
  echo "  fs_jobs    : $FS_JOBS"
  echo "  fs_outdir  : $FS_OUTDIR"
  echo "──────────────────────────────────────────────────────────"

  FS_CONV_FILES=()
  if [ -n "$CONV_FILE" ]; then
    if [ ! -f "$CONV_FILE" ]; then
      echo "[FaithScore] Error: --conv file not found: $CONV_FILE"
      exit 1
    fi
    FS_CONV_FILES=("$CONV_FILE")
  elif [ -n "$INPUT_DIR" ] && [ -d "$INPUT_DIR" ]; then
    for f in "$INPUT_DIR"/*.json; do
      [ -f "$f" ] && FS_CONV_FILES+=("$f")
    done
  else
    echo "[FaithScore] Error: provide --conv <file> or --input_dir <dir>."
    exit 1
  fi

  if [ "${#FS_CONV_FILES[@]}" -eq 0 ]; then
    echo "[FaithScore] No JSON files found."
    exit 1
  fi

  FS_EVAL_ARGS=()
  FS_EVAL_ARGS+=(--vem_type "$FS_VEM_TYPE")
  FS_EVAL_ARGS+=(--llava_path "$FS_LLAVA_PATH")
  FS_EVAL_ARGS+=(--openai_model "$FS_OPENAI_MODEL")
  FS_EVAL_ARGS+=(--openai_base_url "$FS_OPENAI_BASE_URL")

  fs_run_batch() {
    local idx=$1 conv=$2 per_job=$3 total=$4 conv_outdir=$5
    local start=$(( idx * per_job ))
    [ $start -ge "$total" ] && return 0
    local count=$per_job
    [ $((start + count)) -gt "$total" ] && count=$((total - start))
    local out_path="${conv_outdir}/batch_${idx}.json"
    echo "    [FaithScore] batch $idx: samples $start-$((start+count-1)) -> $out_path"
    "$PYTHON" "$FS_SCRIPT" --mode eval \
      --conv "$conv" \
      --start_idx "$start" \
      --sample_num "$count" \
      --save_judgments "$out_path" \
      "${FS_EVAL_ARGS[@]}"
  }

  fs_run_one_conv() {
    local conv="$1"
    local base
    base=$(basename "$conv" .json)
    local conv_outdir="${FS_OUTDIR}/${base}"
    mkdir -p "$conv_outdir"

    echo "  [FaithScore] $conv -> $conv_outdir"

    local total
    total=$("$PYTHON" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$conv" 2>/dev/null || echo 0)
    if [ "$total" -eq 0 ]; then
      echo "  [FaithScore] Warning: $conv has 0 samples, skipping."
      return 0
    fi

    local per_job
    per_job=$(( (total + FS_NUM_JOBS - 1) / FS_NUM_JOBS ))
    local actual_jobs
    actual_jobs=$(( (total + per_job - 1) / per_job ))

    if [ "$FS_JOBS" -le 1 ]; then
      for i in $(seq 0 $((actual_jobs - 1))); do
        fs_run_batch "$i" "$conv" "$per_job" "$total" "$conv_outdir"
      done
    else
      for i in $(seq 0 $((actual_jobs - 1))); do
        while [ "$(jobs -r 2>/dev/null | wc -l)" -ge "$FS_JOBS" ]; do sleep 0.5; done
        ( fs_run_batch "$i" "$conv" "$per_job" "$total" "$conv_outdir" ) &
      done
      _fs_fail=0
      for _job in $(jobs -p); do wait "$_job" || _fs_fail=$?; done
      [ "$_fs_fail" -ne 0 ] && echo "[FaithScore] Warning: one or more parallel batches failed (last exit: $_fs_fail)"
    fi

    if [ "$FS_MERGE" = true ]; then
      local judgment_files=()
      for i in $(seq 0 $((actual_jobs - 1))); do
        local bf="${conv_outdir}/batch_${i}.json"
        [ -f "$bf" ] && judgment_files+=("$bf")
      done
      if [ "${#judgment_files[@]}" -gt 0 ]; then
        echo "  [FaithScore] Merging ${#judgment_files[@]} batches -> ${conv_outdir}/merged_all.json"
        "$PYTHON" "$FS_SCRIPT" --mode merge \
          --judgment_files "${judgment_files[@]}" \
          --merge_output "${conv_outdir}/merged_all.json"
      fi
    fi
  }

  fs_total_count=0
  for conv in "${FS_CONV_FILES[@]}"; do
    fs_run_one_conv "$conv"
    ((fs_total_count++)) || true
  done

  echo "[FaithScore] Done: $fs_total_count file(s) processed -> $FS_OUTDIR"
fi


# ═════════════════════════════════════════════════════════════════
#  HaELM
# ═════════════════════════════════════════════════════════════════
if [ "$RUN_HE" = true ]; then
  echo ""
  echo "──────────────────────────────────────────────────────────"
  echo " HaELM"
  echo "  llama_path : $HE_LLAMA_PATH"
  echo "  checkpoint : $HE_CHECKPOINT"
  echo "  sample_num : $HE_SAMPLE_NUM"
  echo "  outdir     : $HE_OUTDIR"
  echo "  jobs       : $JOBS"
  echo "──────────────────────────────────────────────────────────"

  HE_CONV_FILES=()
  if [ -n "$CONV_FILE" ]; then
    if [ ! -f "$CONV_FILE" ]; then
      echo "[HaELM] Error: --conv file not found: $CONV_FILE"
      exit 1
    fi
    HE_CONV_FILES=("$CONV_FILE")
  elif [ -n "$INPUT_DIR" ] && [ -d "$INPUT_DIR" ]; then
    for f in "$INPUT_DIR"/*.json; do
      [ -f "$f" ] && HE_CONV_FILES+=("$f")
    done
  else
    echo "[HaELM] Error: provide --conv <file> or --input_dir <dir>."
    exit 1
  fi

  if [ "${#HE_CONV_FILES[@]}" -eq 0 ]; then
    echo "[HaELM] No JSON files found."
    exit 1
  fi

  mkdir -p "$HE_OUTDIR"

  # Per-file runner — one Python process per file.
  # Captures output to parse printed hallucination rates for the summary table.
  he_run_one() {
    local conv="$1"
    local base
    base=$(basename "$conv" .json)
    local outfile="${HE_OUTDIR}/${base}_haelm.json"
    local log_file="${HE_OUTDIR}/.haelm_log_${base}.txt"
    echo "  [HaELM] $conv -> $outfile"

    "$PYTHON" "$HE_SCRIPT" \
      --conv "$conv" \
      --llama_path "$HE_LLAMA_PATH" \
      --checkpoint_path "$HE_CHECKPOINT" \
      --outfile "$outfile" \
      --sample_num "$HE_SAMPLE_NUM" \
      2>&1 | tee "$log_file"

    # Extract printed rates and write summary fragment
    "$PYTHON" - "$log_file" "$conv" "$outfile" \
        "$HE_OUTDIR/.haelm_summary_${base}.json" <<'PYEOF'
import json, sys, os, re
log_path, in_path, out_path, sum_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sentence_rate, sample_rate = None, None
try:
    for line in open(log_path, encoding='utf-8'):
        m = re.search(r'Sentence level hallucination rate:\s*([\d.]+)', line)
        if m:
            sentence_rate = float(m.group(1))
        m = re.search(r'Sample level hallucination rate:\s*([\d.]+)', line)
        if m:
            sample_rate = float(m.group(1))
except Exception:
    pass
rec = {
    'input_file':           os.path.basename(in_path),
    'output_file':          os.path.basename(out_path),
    'sentence_halluc_rate': sentence_rate,
    'sample_halluc_rate':   sample_rate,
}
with open(sum_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(rec, ensure_ascii=False))
os.remove(log_path)
PYEOF
  }

  he_count=0
  # NOTE: each HaELM instance loads a full LLaMA model.
  # Use -j >1 only if you have sufficient GPU memory for concurrent loads.
  if [ "$JOBS" -le 1 ]; then
    for conv in "${HE_CONV_FILES[@]}"; do
      he_run_one "$conv"
      ((he_count++)) || true
    done
  else
    for conv in "${HE_CONV_FILES[@]}"; do
      while [ "$(jobs -r 2>/dev/null | wc -l)" -ge "$JOBS" ]; do sleep 0.5; done
      ( he_run_one "$conv" ) &
      ((he_count++)) || true
    done
    _he_fail=0
    for _job in $(jobs -p); do wait "$_job" || _he_fail=$?; done
    [ "$_he_fail" -ne 0 ] && echo "[HaELM] Warning: one or more parallel jobs failed (last exit: $_he_fail)"
  fi

  # Merge HaELM summary fragments into a single table
  if [ "$he_count" -gt 0 ]; then
    he_summary_count=$(find "$HE_OUTDIR" -maxdepth 1 -name '.haelm_summary_*.json' 2>/dev/null | wc -l || true)
    if [ "$he_summary_count" -gt 0 ]; then
      "$PYTHON" - "$HE_OUTDIR" <<'PYEOF'
import json, sys, os, glob
outdir = sys.argv[1]
recs = []
for p in sorted(glob.glob(os.path.join(outdir, '.haelm_summary_*.json'))):
    try:
        recs.append(json.loads(open(p, encoding='utf-8').read()))
    except Exception:
        pass
with open(os.path.join(outdir, 'haelm_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(recs, f, ensure_ascii=False, indent=2)
with open(os.path.join(outdir, 'haelm_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('input_file\toutput_file\tsentence_halluc_rate\tsample_halluc_rate\n')
    for r in recs:
        f.write('{}\t{}\t{}\t{}\n'.format(
            r['input_file'], r['output_file'],
            r.get('sentence_halluc_rate', ''), r.get('sample_halluc_rate', '')))
for p in glob.glob(os.path.join(outdir, '.haelm_summary_*.json')):
    try: os.remove(p)
    except Exception: pass
PYEOF
      echo ""
      echo "[HaELM] Summary: $HE_OUTDIR/haelm_summary.json"
      cat "$HE_OUTDIR/haelm_summary.txt"
    fi
  fi
  echo "[HaELM] Done: $he_count file(s) processed -> $HE_OUTDIR"
fi


# ═════════════════════════════════════════════════════════════════
#  SoftSPICE
# ═════════════════════════════════════════════════════════════════
if [ "$RUN_SP" = true ]; then
  # Default: if SG also ran, score the SG outputs (they carry unique_sg).
  # Otherwise fall back to --input_dir (which must already contain unique_sg).
  if [ -z "$SP_INDIR" ]; then
    if [ "$RUN_SG" = true ] && [ -d "$SG_OUTDIR" ]; then
      SP_INDIR="$SG_OUTDIR"
    else
      SP_INDIR="$INPUT_DIR"
    fi
  fi

  if [ -z "$SP_INDIR" ] || [ ! -d "$SP_INDIR" ]; then
    echo "[SoftSPICE] Error: --spice_dir or --input_dir must be a valid directory (got: '${SP_INDIR}')."
    exit 1
  fi

  echo ""
  echo "──────────────────────────────────────────────────────────"
  echo " SoftSPICE"
  echo "  input_dir  : $SP_INDIR"
  echo "  text_enc   : $SP_TEXT_ENCODER"
  echo "  metric     : $SP_METRIC"
  echo "──────────────────────────────────────────────────────────"

  "$PYTHON" "$SP_SCRIPT" \
    --sg_dir "$SP_INDIR" \
    --text_encoder "$SP_TEXT_ENCODER" \
    --metric "$SP_METRIC"

  echo "[SoftSPICE] Done -> $SP_INDIR/spice_summary.json"
fi


echo ""
echo "============================================================"
echo " All requested graders finished."
[ "$RUN_SG" = true ] && echo "  SG results        : $SG_OUTDIR"
[ "$RUN_FS" = true ] && echo "  FaithScore results: $FS_OUTDIR"
[ "$RUN_HE" = true ] && echo "  HaELM results     : $HE_OUTDIR"
[ "$RUN_SP" = true ] && echo "  SoftSPICE summary : $SP_INDIR/spice_summary.json"
echo "============================================================"
