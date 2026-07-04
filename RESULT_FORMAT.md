# Result Format — handoff doc for the analysis agent

Everything an analyst needs to read the JSON outputs without re-discovering the schema. Last revised by the run-orchestrator agent after running v19 on VG (100 samples) + SVG-500, the v19.5/v19.6mllm experiments, and the POPE (DSG_v2 → DSG_v4) pipeline on gpt-4o transcripts.

---

## 1. Top-level directory layout

```
work_dirs/
├── vg/
│   ├── final_run_v18_gpt4o_resume5_completed/   # v18 (5 contexts) full VG-100 — canonical v18 outputs
│   ├── final_run_v18_gpt4o_completed/           # earlier v18 (less complete, mixed)
│   ├── ablation_baseline2/                       # v18 baseline (dyna_conv.py, max_rounds=10) — used as VG-side caption baseline
│   ├── v19/                                      # v19 (2 contexts compact) local-VLM runs (caches)
│   ├── v19_api/                                  # v19 API runs + POPE artifacts (gpt-4o etc.)
│   ├── v19_api_completed/                        # frozen / past completed
│   ├── v19d5/                                    # v19.5 (no GT for regular/follow-up)  — experiment, not "good"
│   ├── v19d6mllm_test/                           # v19.6mllm (multimodal examiner) — 5-sample TEST
│   ├── final_run_v20_gpt4o_small[_completed]/    # v20
│   ├── caption/, caption_api/, caption_contextualized/  # caption-only baselines
│   └── vg_image_cache/                           # auto-populated PIL cache for VG URLs
└── svg/
    ├── ablation_baseline2/                       # v18-equivalent SVG baseline (300+ entries each, mostly cache-only)
    ├── caption/, caption_api/                    # caption baselines for SVG
    └── v19/                                      # v19 SVG-500 — 7 models (6 done, opera still running)
```

`work_dirs/` is gitignored. Big caches: `utils/.vg_cache/` (2.2 GB raw VG JSON), `vg_image_cache/`.

---

## 2. Run-version semantics

| version | context-gen LLM | conv LLM | contexts/image | rounds budget | Notes |
|---|---|---|---|---|---|
| **v18** (`dyna_conv_v18resume5.py`) | gpt-5 | gpt-4o | 5 | max 20, mean ~10 | Canonical baseline; `final_run_v18_gpt4o_resume5_completed/` |
| **v18 baseline** (`dyna_conv.py`, `--max_rounds 10`) | none (text-only conv) | — | — | 10 | aka "ablation_baseline2"; not a v18 dyna run, single-turn loop |
| **v19** (`dyna_conv_v19.py`) | gpt-5 | gpt-4o | 2 | 20, mean ~12 | Compact VG format (~58% fewer scene-graph tokens) |
| **v19.5** (`dyna_conv_v19d5.py`) | gpt-5 | gpt-4o | 2 | 20 | Same as v19 but **no LLM-generated GT** for regular/follow-up rounds (gt = "" for those q_types). User concluded "not good", abandoned. |
| **v19.6mllm** (`dyna_conv_v19d6mllm.py`) | gpt-5 | gpt-4o | 2 | 20 | Same as v19 but the conv LLM gets the actual **image** attached to the first user turn. Test only. |
| **v20** (`dyna_conv_v20.py`) | gpt-5 | gpt-4o | 2 | varies | Adds `phase15_subset_object_ids`. |
| **v19_api** | gpt-5 | gpt-4o | 2 | 20 | examinee is an API model (gpt-4o, gpt-5.4-mini, glm-5v-turbo). |

**SVG datasets** are loaded from HF (`Icey444/svg500_in_vg`) and have images pre-attached at `case["image"]`. **VG datasets** are loaded from raw VG JSON (`utils/vg.py`) which does **not** attach images — must fetch via URL (see §6).

---

## 3. Per-sample (entry) schema — dyna runs

A "sample" in a dyna-run output is one **(image, context)** pair. So an image with 5 contexts produces 5 entries. Output JSON is a flat list of these.

```json
{
  "image_id": 1,                                  // VG id (int) or SVG id (string like "1159399.jpg")
  "url": "https://cs.stanford.edu/.../1.jpg",     // VG image URL
  "width": 800, "height": 600,
  "coco_id": null, "flickr_id": null,
  "sg": { "objects": {...}, "relationships": [...], "regions": [...] },  // scene graph
  "metadata": { "objects": [...], "relationships": [...], ... },        // raw VG dump per source key
  "context": {
    "background": "Late morning on a brick sidewalk downtown ...",
    "goal":       "Reach the orange parking meter, confirm ...",
    "relevant_objects": ["instance_24", "instance_18", ...]              // object_ids
  },
  "relevant_objects": [...],                                              // duplicate of context.relevant_objects
  "phase15_subset_object_ids": [...],                                     // v20 only
  "conversations": [ ... see §4 ... ]
}
```

**Key invariant**: an image_id can appear ≥1 time (once per context). Final JSONs sometimes have **fewer** total entries than `num_samples × contexts/image` because context-generation can fail (3 retries, then sample skipped). Typical loss: 0–4 entries per 1000.

---

## 4. `conversations` schema — list of round dicts

Each turn corresponds to one examiner question + one VLM response.

```json
{
  "round_id": 7,
  "prompt": "What color is the dog sitting near the man?",   // examiner's question to VLM
  "response": "the dog sitting next to the bench is black.", // VLM's answer (lowercased)
  "q_type": "unanswerable",                                  // regular | follow-up | adversarial | unanswerable | end
  "gt": "I can't answer the question because the object doesn't exist.",  // examiner's claimed GT (see §4.1)
  "meta_msg": [...]                                          // present only for adversarial/unanswerable; multi-step gen log
}
```

### 4.1 q_type semantics + GT convention

| q_type | what the examiner is doing | source of `gt` field |
|---|---|---|
| `regular` | Asks a normal question grounded in `relevant_objects` + context | LLM-generated (in v18/v19), **""** in v19.5 |
| `follow-up` | Challenges/probes the previous VLM answer | LLM-generated (in v18/v19), **""** in v19.5 |
| `adversarial` | Asks about something **plausibly absent** that co-occurs with visible content | **Always hardcoded `"No"`** |
| `unanswerable` | Presupposes a nonexistent entity (false-premise trap) | **Always hardcoded `"I can't answer the question because the object doesn't exist."`** |
| `end` (type_id=5) | Termination signal; no `prompt`/`response` saved | n/a |

**Distribution observed (v18 VG)**: ~30% regular, ~26% follow-up, ~32% adversarial, ~12% unanswerable.

---

## 5. POPE pipeline output (DSG_v2 → DSG_v4)

These extra fields show up after running the POPE pipeline on a dyna output.

### 5.1 After `examiner/DSG_v2.py` → `<name>_pope_converted.json`

Each `conversations[turn]` gets a new `dsg_qa` array — atomic yes/no questions extracted from that turn's response by Qwen3-30B:

```json
"dsg_qa": [
  {
    "index": 1,
    "question": "Is there a parking ticket machine?",
    "answer": "No",                  // extracted from transcript ("does the response say yes or no?")
    "has_answer": true,
    "evidence": "Turn 2 states: 'i can't specifically see ...'",
    "tuple": "entity - whole"
  },
  ...
]
```

A turn can spawn **3–20** POPE questions. `answer` is what Qwen extracted from the original response transcript (not necessarily ground truth — see §5.3).

### 5.2 After `examiner/DSG_v4.py --verify_only` → `<name>_with_both_answers.json`

Each `qa` gets:

| field | meaning |
|---|---|
| `gt_answer`, `gt_reasoning` | Canonical ground truth: Qwen judges against the **VG annotation/scene graph**, not the transcript. This is the most reliable label, but Qwen still gets fooled on adv/una when the transcript confidently asserts the absent thing — see §5.3. |
| `dynamic_response`, `dynamic_is_correct` | Qwen's yes/no extraction from the original v19 conversation response (mode = "did the original VLM's answer imply Yes/No for this Q?") |
| `single_response`, `single_response_raw`, `single_is_correct` | **As of 2026-05-04, this is the ISOLATED answer**: gpt-4o was re-asked the single question with just the image (1 call per Q). |
| `single_response_batch`, `single_response_batch_raw`, `single_batch_is_correct` | **Legacy** — the original batched call (whole turn's POPE Qs in one VLM call, "Yes/No per line"). Kept for backwards-compat. |

Earlier files (before the rename) had `single_response` = batched and `single_response_isolated` = isolated. Files with `single_response_batch` already-present indicate the swap is done.

### 5.3 Canonical GT convention (CRITICAL)

The Qwen-extracted `gt_answer` is **not** trustworthy for adversarial/unanswerable q_types: when the original VLM hallucinated a "Yes" for an absent object, the transcript text says yes, and Qwen faithfully extracts that as the GT. This **inflates accuracy** by ~15 pp overall and ~50 pp on adv/una individually.

**Canonical fix** (already applied in `_isolated_all.json` and `_isolated20.json`): force `gt_answer = "No"` for every `q_type ∈ {adversarial, unanswerable}`. `gt_reasoning` for those entries says: `"POPE-canonical: <q_type> questions are designed to be unanswerable/absent; GT forced to No regardless of transcript-extracted answer."`

When computing accuracy, **always** check whether the file has been canonical-flipped (look at any adv qa: if `gt_answer == "No"` and `gt_reasoning` starts with "POPE-canonical:", you're on canonical ground). Otherwise apply the rule yourself in code:

```python
def gt_canonical(qa, q_type):
    if q_type in ("adversarial", "unanswerable"):
        return "No"
    s = (qa.get("gt_answer") or qa.get("answer") or "").strip().lower()
    return "Yes" if s.startswith("y") else ("No" if s.startswith("n") else None)
```

---

## 6. Image loading (VG vs SVG)

| dataset | how images are attached |
|---|---|
| **SVG** | `load_svg()` returns samples with `case["image"]` already set to a `PIL.Image` (HF dataset has `image` column). |
| **VG** | `load_vg()` does NOT set `case["image"]`. The sample has `image_url` (sometimes empty string due to a bug — see below). The canonical URL lives in `utils.vg._image_data_raw[i]["url"]`. |

VG image fetcher pattern (used by `dyna_conv_v19d6mllm.py` and `caption_contextualized.py`):

```python
from utils.vg import _image_data_raw
url_map = {r["image_id"]: r["url"] for r in _image_data_raw}
# cache to work_dirs/vg_image_cache/<image_id>.jpg, then PIL.Image.open(...).convert("RGB")
```

**Bug to be aware of**: `_build_objects_compat` in `utils/vg.py` does `rec.get("url", "")` but the field name is `image_url`. So `case["image_url"]` is empty. Use `_image_data_raw` directly.

---

## 7. Reading accuracy: dyn vs iso vs bat

For a 200-sample run, computing accuracy looks like this (assumes the canonical-flipped file):

```python
import json
from collections import defaultdict

d = json.load(open("work_dirs/vg/v19_api/openai_gpt-4o_with_both_answers_isolated_all.json"))

def yn(s):
    if not s: return None
    s = str(s).strip().lower()
    return "Yes" if s.startswith("y") else ("No" if s.startswith("n") else None)

stats = defaultdict(lambda: {"n":0,"dyn":0,"iso":0,"bat":0})
for sample in d:
    for turn in sample.get("conversations", []):
        qt = turn.get("q_type","?")
        for qa in turn.get("dsg_qa", []):
            gt = yn(qa.get("gt_answer") or qa.get("answer"))
            if gt is None: continue
            for k in (qt, "ALL"):
                s = stats[k]; s["n"] += 1
                if qa.get("dynamic_response") == gt: s["dyn"] += 1
                if qa.get("single_response") == gt:        s["iso"] += 1   # isolated, primary
                if qa.get("single_response_batch") == gt:  s["bat"] += 1   # legacy batched
```

### Headline numbers (v19_api gpt-4o, 200 VG samples, canonical):

| q_type | n_gt | dyn | iso | bat |
|---|---|---|---|---|
| ALL | 15,626 | 68.25% | 68.93% | 67.52% |
| regular | 5,709 | 86.34% | 83.82% | 79.24% |
| follow-up | 4,780 | 84.39% | 84.90% | 81.38% |
| adversarial | 3,931 | 35.87% | 34.44% | 38.41% |
| unanswerable | 1,206 | 24.13% | 47.60% | 51.91% |

**Reads**:
- All three modes are within ~1 pp on ALL (~68%) — basically tied.
- **dyn wins on regular/follow-up** because the original conversational response captured these well; re-asking in isolation drops a few pp.
- **bat actually wins on adv/una** (counterintuitive!) because in long batched lists gpt-4o tends to insert "No" to break monotony, which happens to match the canonical "No". Don't read this as "batched is better" — it's an anchoring artifact.
- **dyn collapses on adv (35.87) and una (24.13)** — this is the hallucination signal: the original VLM played along with the false premise of the leading question.
- The pre-canonical numbers were ~83% on ALL and ~83% on adv/una — **inflated by ~15–50 pp**. Always verify canonical when reporting accuracy.

---

## 8. POPE call-mode default (since 2026-05-04)

`examiner/DSG_v4.py` now does **isolated by default** (1 VLM call per question, parallelized through `vlm_workers=36`). To opt back into the old per-turn batched call:

```bash
python examiner/DSG_v4.py ... --vlm_batch_per_turn
```

The legacy batched mode caused ~12% of POPE answers to flip from the isolated answer (long-batch list-anchoring bias). Default isolated recovers ~2 pp on overall accuracy and ~4–6 pp on follow-up/adversarial.

---

## 9. API endpoints in use (and retired)

| service | endpoint | auth env | notes |
|---|---|---|---|
| **uniapi** (default for examiner LLMs + VLM API) | `https://api.uniapi.io/v1` | `OPENAI_API_KEY` (set to uniapi key) + `OPENAI_BASE_URL` | OpenAI-compatible. Use `uniapi/<model>` model_path prefix in DSG_v4 to auto-route. |
| **External Qwen3-30B** (DSG_v2 + DSG_v4 verify) | `http://109.61.17.115:8000/v1` | `Authorization: Bearer william` | Set `REMOTE_API_URL`, `REMOTE_API_KEY=william`, `REMOTE_API_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507`. DSG_v4 also reads these (we patched it 2026-05-03). |
| **Local Qwen3-30B vLLM** (`scripts/host_qwen3_30b_gpu*.sh`) | `http://localhost:8004/v1` | none | Use `--local` in DSG_v2 / default in DSG_v4 (when REMOTE_* not set). Uses GPUs 0,1 by default (host script needs `CUDA_HOME=/usr/local/cuda`, `--gpu-memory-utilization 0.4`). |
| **Harbor / Parity** | ~~`http://pp-api-...:3000/v1`~~ | — | **RETIRED.** Hard-guarded in `infer/api_vlm.py` and `examiner/DSG_v4.py`. Don't reuse. |

`utils/llm.py` automatically injects `extra_body={"reasoning_effort": "minimal"}` for `gpt-5*`, `o1*`, `o3*`, `o4*` models — without it, gpt-5 burns the entire `max_tokens` on hidden reasoning and returns empty content.

---

## 10. File naming conventions

| pattern | meaning |
|---|---|
| `<model>.json` | Final output. Only written when the run completes; until then the cache is the live artifact. |
| `<model>_cache.json` | Incremental cache. Written after every image. Used for resume (the run skips images already cached). |
| `<model>_pope_converted.json` | After DSG_v2 — adds `dsg_qa` to each turn. Same length as input. |
| `<model>_with_both_answers.json` | After DSG_v4 — adds dyn/iso/bat fields per qa. |
| `<model>_with_both_answers_isolated_all.json` | DSG_v4 with isolated as primary `single_response` across all 200 samples (the canonical analysis file as of 2026-05-04). Same shape, different population strategy. |
| `<model>_pope_output.json`, `<model>_with_both_answers.json`, `<model>_extracted.json` | Earlier pope-pipeline artifacts; superseded. |
| `hallucinated_words_<model>.json` | Output of the CHAIRi grader. |
| `mmhal_<model>.json` | Output of the MMHal grader. |
| `<model>_sg_ged.json`, `<model>_sg_delta_con.json` | Scene-graph-distance graders (GED, ΔCon). |
| `<model>_haelm.json` | HaELM grader output. |
| `<model>/softspice/spice_summary.json` | SoftSPICE grader output. |
| `<model>/all_metrics.json` | Aggregated grader output (per-metric scores). |
| `chair_results_summary.csv` | CHAIRi summary across all models in a directory. |

`<model>_cache.json` and `<model>.json` typically have **identical content** once the run finishes (the final write copies the cache). Don't compute twice.

---

## 11. Known incidents and gotchas

1. **OOM cascade on shared GPUs** — multiple v19 SVG procs were SIGKILLed on 2026-05-02 ~02:50 UTC because GPU 6/7 hit memory pressure from other users' processes. Always check `nvidia-smi` for free room before launching, and prefer GPUs 0–4 (allocated to us) when possible.

2. **OPENAI_API_KEY=Harbor was leaked-then-revoked**. Use the live uniapi key (set via env var, do **not** hardcode); `scripts/cmds.sh` is gitignored for this reason. Pull it from your project secret store.

3. **scripts/cmds.sh is a runbook, NOT just env exports**. Sourcing it accidentally kicks off downstream training scripts. To pull just the API key: `eval "$(grep -E '^export OPENAI_API_KEY=' scripts/cmds.sh | head -n1)"`.

4. **DSG_v4 used to hardcode Qwen at `localhost:8004`** — patched 2026-05-03 to read `REMOTE_API_URL`/`REMOTE_API_KEY`/`REMOTE_API_MODEL` first.

5. **gpt-5 empty content** — without `reasoning_effort=minimal`, gpt-5 spends all of `max_tokens` on hidden reasoning tokens. Caused failed v19.5 runs early on. Already patched in `utils/llm.py`.

6. **`utils/vg.py` `image_url` bug**: `_build_objects_compat` reads the wrong field name. Use `_image_data_raw[i]["url"]` directly.

7. **Some final JSONs have ~996/1000 entries instead of 1000**. That's normal — context-generation occasionally fails after 3 retries; sample is silently skipped. Look at the cache count to confirm where the run stalled.

8. **Two contexts/image in v19, five in v18**. When pooling across versions, the per-image entry count differs.

---

## 12. Code locations (quick map)

| file | purpose |
|---|---|
| `examiner/dyna_conv_v18resume5.py` | v18 examiner (5 contexts) |
| `examiner/dyna_conv_v19.py` | v19 examiner (2 contexts, compact) |
| `examiner/dyna_conv_v19d5.py` | v19.5 (no GT for reg/foll) — abandoned |
| `examiner/dyna_conv_v19d6mllm.py` | v19.6mllm (multimodal examiner) |
| `examiner/dyna_conv_v20.py` | v20 |
| `examiner/caption.py`, `examiner/caption_contextualized.py` | caption baselines |
| `examiner/DSG_v2.py` | POPE step 1 — extract yes/no Qs from transcript via Qwen |
| `examiner/DSG_v4.py` | POPE step 2/3 — VLM re-ask (isolated by default, `--vlm_batch_per_turn` to revert) + Qwen verify_only |
| `examiner/dsg_v4_isolated_redo.py` | one-shot script: re-run DSG_v4 in isolated mode, write `single_response_isolated*` fields side-by-side |
| `utils/llm.py` | `LLMChat` wrapper (auto-injects `reasoning_effort=minimal` for reasoning models) |
| `infer/api_vlm.py` | API VLM providers (OpenAI/Uniapi/Gemini/Zhipu) — Parity is hard-guarded retired |
| `infer/loader.py` | local-model loader (matches model_path → eval function) |
| `utils/vg.py`, `utils/svg.py`, `utils/coco.py` | dataset loaders |
| `scripts/dyna-v19/v19_svg_main.sh` | canonical SVG-500 launcher (7 models) |
| `scripts/pope/extract_infer_and_eval.sh` | wrapper that chains DSG_v2 → DSG_v4 (auto-hosts vLLM for VLM step; doesn't support API VLMs natively — bypass for gpt-4o) |
| `scripts/host_qwen3_30b_gpu*.sh` | local Qwen3-30B vLLM host scripts (multiple GPU configs) |

---

## 13. The 4 things to remember

1. **Canonical GT** for adversarial/unanswerable is **always "No"**. Don't trust `gt_answer` extracted from transcripts on those types.
2. **`single_response` is now isolated** (1 Q/call). The legacy batched value is in `single_response_batch`. Pre-2026-05-04 files have it the other way around.
3. **`work_dirs/vg/v19_api/openai_gpt-4o_with_both_answers_isolated_all.json`** is the canonical 200-sample POPE-pipeline artifact (canonical GT applied, isolated as primary).
4. **VG samples don't have `case["image"]`** — fetch from URL (`utils.vg._image_data_raw`) and cache in `work_dirs/vg_image_cache/`.
