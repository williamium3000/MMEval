# Setup guide for context-eval-mllm

End-to-end instructions to set up a fresh machine and run a v19ban2type VG eval. Tested on Ubuntu 22.04 with 8× H200 GPUs.

---

## 1. Prerequisites

- **OS**: Linux (Ubuntu 20.04+ or similar). Tested on 22.04.
- **CUDA**: 12.x toolkit (`nvcc` at `/usr/local/cuda` or similar). `nvidia-smi` reports the driver runtime.
- **Python**: 3.10–3.12 (we use 3.10 for most envs; 3.12 for the `verl` env).
- **Conda / Miniconda**: required for the per-model environments.
- **Disk**: ~50 GB for code + envs + dataset caches; outputs grow per run.
- **API keys**:
  - `OPENAI_API_KEY` — uniapi key (acts as both OpenAI-compat and Gemini-native key on uniapi). Set `OPENAI_BASE_URL=https://api.uniapi.io/v1`.
  - `HF_TOKEN` — HuggingFace user token; needed for gated repos (e.g., `google/gemma-3-12b-it`).

---

## 2. Clone and basic layout

```bash
git clone https://github.com/williamium3000/context-eval-mllm.git
cd context-eval-mllm
git checkout develop-v2
mkdir -p work_dirs tmp
```

The repository expects `work_dirs/` for outputs (gitignored) and `tmp/` for vLLM logs. Sub-trees you'll see populated as runs progress:

```
work_dirs/
├── vg/<run_name>/<model>.json[+ _cache.json]   # dyna outputs
├── svg/<run_name>/<model>.json
├── envs/                                       # per-model conda envs (created below)
└── vg_image_cache/                             # auto-populated PIL cache
```

See `RESULT_FORMAT.md` for output schema details.

---

## 3. Conda environments (per model family)

Each model family has its own env because library version pinning conflicts. Create them under `work_dirs/envs/` so they live next to the code (and stay out of the global conda root).

```bash
# Conda hook
eval "$(conda shell.bash hook)"

# 1. qwenvl — Qwen2.5-VL-* family
conda create -p work_dirs/envs/qwenvl python=3.10 -y
conda activate work_dirs/envs/qwenvl
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.49,<4.55" accelerate "qwen-vl-utils[decord]" \
            openai requests tqdm pillow nltk fsspec aiohttp datasets pycocotools \
            "huggingface_hub<1.0"
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# 2. qwenvl3 — Qwen3-VL-*, also used as the LLaVA-1.5 host (the local llava env has stale torch)
conda create -p work_dirs/envs/qwenvl3 python=3.10 -y
conda activate work_dirs/envs/qwenvl3
pip install --upgrade pip
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=4.57" accelerate "qwen-vl-utils[decord]" \
            openai requests tqdm pillow nltk fsspec aiohttp datasets pycocotools \
            "huggingface_hub<1.0"
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# 3. internvl — InternVL2 / InternVL2_5 / InternVL3
conda create -p work_dirs/envs/internvl python=3.10 -y
conda activate work_dirs/envs/internvl
pip install --upgrade pip
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.37.2" "accelerate==0.34.2" \
            openai requests tqdm pillow timm nltk fsspec aiohttp datasets pycocotools \
            sentencepiece einops "huggingface_hub<1.0"
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# 4. gemma3 — gemma-3-* family (gated; needs HF_TOKEN)
conda create -p work_dirs/envs/gemma3 python=3.10 -y
conda activate work_dirs/envs/gemma3
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.50" accelerate \
            openai requests tqdm pillow nltk fsspec aiohttp datasets pycocotools \
            "huggingface_hub<1.0"
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# 5. opera — global conda env for the bundled Opera/MiniGPT-4 codebase (not under work_dirs/envs)
conda create -n opera python=3.10 -y
conda activate opera
pip install --upgrade pip
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install "transformers==4.37.2" accelerate openai requests tqdm pillow nltk \
            timm einops sentencepiece omegaconf hydra-core matplotlib opencv-python \
            "huggingface_hub<1.0"
# Opera additionally needs minigpt4/llava bundled in the repo; the loader resolves
# them via PYTHONPATH (handled by the launcher scripts).

# 6. verl — for vLLM-hosted Qwen3-30B (used as POPE extractor + grader judge).
#    Optional unless you need a local Qwen3-30B server (see §6).
conda create -n verl python=3.12 -y
conda activate verl
pip install --upgrade pip
pip install "vllm>=0.7" "transformers>=4.49" openai requests tqdm
```

The exact pin versions matter most for the InternVL env (transformers 4.37.2 + accelerate 0.34.2 + torch 2.9.1). Other envs are more forgiving but lock pytorch / transformers compat.

> **Gotcha**: any env where `_image_data_raw` (`utils/vg.py`) is imported needs `nltk` + the wordnet corpus, even though it isn't used for VG runs — `utils/svg.py` imports it at module load. The `pip install nltk` + `nltk.download('wordnet')` lines above handle that.

---

## 4. Local checkpoints (only if you'll run Opera)

```bash
mkdir -p data/checkpoints/opera
# Place Opera's llava-1.5 checkpoint at:
#   data/checkpoints/opera/llava-1.5/{config, weights, etc.}
# (Symlink from a shared location is fine.)
```

For HF-hosted models (Qwen, InternVL, gemma, llava-hf), the env will download on first use. Set `HF_HOME=/path/with/space` if `~/.cache/huggingface` is small.

---

## 5. Datasets

- **VG (Visual Genome)**: `utils/vg.py` downloads the 5 raw JSON dumps from the official VG site to `utils/.vg_cache/` on first use. ~2.2 GB; expect a one-time 5–10 minute download. Images are pulled per-`image_id` lazily into `work_dirs/vg_image_cache/`.
- **SVG**: loaded from HF dataset `Icey444/svg500_in_vg` on first use (images included in the dataset).

No manual downloads required — first run handles both.

---

## 6. Optional: local Qwen3-30B vLLM (POPE extractor / grader judge)

The POPE pipeline (`examiner/DSG_v2.py`, `examiner/DSG_v4.py`) and several graders use Qwen3-30B-A3B-Instruct as a judge. Two options:

**(a) Use an external endpoint** (fastest if available):

```bash
export REMOTE_API_URL="http://<host>:8000/v1"
export REMOTE_API_KEY="<token>"
export REMOTE_API_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
```

**(b) Host locally on 2 GPUs** (the in-repo helper):

```bash
conda activate verl
export CUDA_HOME=/usr/local/cuda
bash scripts/host_qwen3_30b_gpu01.sh
# Defaults: GPUs 0,1; --gpu-memory-utilization 0.45; port 8004.
# If memory is tight, lower the util to 0.4 (edit the script or set GPU_UTIL).
# Wait for the readiness probe (or `curl http://localhost:8004/health`).
```

Then point `--qwen_port 8004` at it (DSG_v4 falls back to localhost:8004 when REMOTE_API_URL isn't set).

---

## 7. Smoke test — single sample

Before launching a real eval, sanity-check the API:

```bash
export OPENAI_API_KEY="<uniapi-key>"
export OPENAI_BASE_URL="https://api.uniapi.io/v1"

eval "$(conda shell.bash hook)"
conda activate work_dirs/envs/qwenvl3
python -c "
import sys; sys.path.insert(0,'.')
from utils.llm import LLMChat
chat = LLMChat(model_name='gpt-5.4-mini', patience=2)
print(chat.chat([{'role':'user','content':'Reply with the single word: pong'}], parser_fn=None))
"
# expect: pong
```

(uniapi gpt-5.x models burn output budget on hidden reasoning by default; `utils/llm.py` already injects `reasoning_effort=minimal` for `gpt-5*`/`o*` — works out of the box.)

---

## 8. Run a v19ban2type VG eval

`v19ban2type` = `v19` with adversarial/unanswerable q_types disabled (only `regular` and `follow-up` are asked). 100 VG samples, 2 contexts/image, max ~20 rounds.

### 8.1 Set environment

```bash
export OPENAI_API_KEY="<uniapi-key>"
export OPENAI_BASE_URL="https://api.uniapi.io/v1"
export HF_TOKEN="<hf-token>"     # only needed for gemma-3-12b-it
# For the gemini-2.5-flash examinee (optional):
export GEMINI_API_KEY="${OPENAI_API_KEY}"
export GEMINI_API_BASE="https://api.uniapi.io/gemini"
```

### 8.2 Pick GPU assignment

`nvidia-smi` to see free memory. 8B examinees need ~25 GB free per GPU (12B gemma needs ~30 GB). Multiple examinees can share one GPU if memory allows.

### 8.3 Launch

```bash
bash scripts/dyna-v19/v19ban2type_vg.sh \
    "llava-1.5-7b-hf=4,InternVL3-8B-Instruct=5,gemma-3-12b-it=6,Qwen2.5-VL-7B-Instruct=7"
```

Supported models (use empty GPU id for the API examinee):

```
llava-1.5-7b-hf            (env: qwenvl3)
InternVL2-8B               (env: internvl)
InternVL2_5-8B             (env: internvl)
InternVL3-8B-Instruct      (env: internvl)
Qwen2.5-VL-7B-Instruct     (env: qwenvl)
gemma-3-12b-it             (env: gemma3)  -- gated, needs HF_TOKEN
opera-llava-1.5            (env: opera)
gemini-2.5-flash=          (no GPU; API via uniapi)
```

Outputs: `work_dirs/vg/v19ban2type/<model>.json` (and `_cache.json`).
Logs: `work_dirs/logs_v19ban2type_vg/<model>.log`.

### 8.4 Monitor

```bash
# Live tqdm
tail -f work_dirs/logs_v19ban2type_vg/<model>.log

# Cached conversations so far
python -c "import json; print(len(json.load(open('work_dirs/vg/v19ban2type/<model>_cache.json'))))"
```

A run is complete when the final `<model>.json` is written; until then the cache is the live artifact. See `RESULT_FORMAT.md` §10 for the file-naming conventions.

### 8.5 Resume a partial run

The launcher's `run_job` skips a model only when both the **final** `<model>.json` exists and the log shows `Results saved to` with no recent errors. To force a resume: leave the `<cache>.json` file in place and re-run the same command — the script will continue from the last completed image.

---

## 9. Verify

After completion (and assuming `dynamic_response`/POPE pipeline isn't needed yet):

```bash
python -c "
import json
d = json.load(open('work_dirs/vg/v19ban2type/llava-1.5-7b-hf.json'))
print(f'entries: {len(d)} (expect 200 = 100 images x 2 contexts; small drop OK)')
print(f'unique image_ids: {len({e[\"image_id\"] for e in d})}')
print(f'rounds in entry 0: {len(d[0][\"conversations\"])}')
qtypes = sorted({c.get(\"q_type\") for e in d for c in e.get(\"conversations\", [])})
print(f'q_types seen: {qtypes}  (expect: regular, follow-up only)')
"
```

A v19ban2type run should report `q_types seen: ['follow-up', 'regular']` — if `adversarial`/`unanswerable` show up, the SWITCH_PROMPT change didn't take effect. The defensive guard in `switch()` should make this impossible.

---

## 10. Common issues

- **`gpt-5*` returns empty content**: `utils/llm.py` already auto-injects `reasoning_effort=minimal` for `gpt-5/o1/o3/o4`. If you see empty replies, confirm the import path isn't overridden by user-site (set `PYTHONNOUSERSITE=1`).
- **CUDA OOM in vLLM** (Qwen3-30B host): lower `--gpu-memory-utilization` to 0.4 or use 4 GPUs at 0.3.
- **`flashinfer / nvcc / cuda_runtime.h` JIT failure**: set `CUDA_HOME=/usr/local/cuda` and prepend `$CUDA_HOME/bin` to `PATH` before launching vLLM. See `RESULT_FORMAT.md` §11 for context.
- **Old shared GPUs cause OOM kills**: use GPUs allocated to your account (`nvidia-smi` first; prefer GPUs with > 30 GB free).
- **`No module named 'nltk'`**: pip install nltk in that env and `nltk.download('wordnet')` (see §3).
- **VG image fetch silently fails**: `utils/vg.py:_build_objects_compat` reads the wrong field key. Use `_image_data_raw[i]['url']` directly (the launchers and `caption_contextualized.py` already do).

See `RESULT_FORMAT.md` for further runtime gotchas and the output-file schema.

---

## 11. Optional: POPE pipeline + graders

After a run finishes, you can run the POPE pipeline (`examiner/DSG_v2.py` → `examiner/DSG_v4.py`) and the standalone graders. Those need a Qwen3-30B server (see §6) plus the same uniapi key. Wrappers live in `scripts/pope/` and `scripts/graders/`. See `RESULT_FORMAT.md` §5 for the resulting field shape.
