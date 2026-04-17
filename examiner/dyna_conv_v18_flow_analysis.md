# Dynamic Conversation Examiner v18 — Analysis

Covers `dyna_conv_v18resume.py` and `dyna_conv_v18resume5.py`.

## Overview

A GPT-powered dynamic multi-round conversation examiner that probes Vision-Language Models (VLMs) for **hallucinations**. GPT acts as the "examiner": it automatically generates contextualised questions, feeds them to a target VLM, and records responses.

## High-Level Flow

1. Load a Visual Genome (VG) image sample with its scene graph (objects, attributes, relations, bounding boxes).
2. **GPT-5** generates **5 diverse first-person contexts** (background + goal + relevant objects) for each image.
3. For each context, **GPT-4o** drives a **multi-round conversation** with the target VLM — asking questions and recording answers.
4. Results are saved per model as JSON.

## Prompt Components

### Phase 1 — Context Generation (GPT-5)

| Prompt | Role |
|--------|------|
| `CONTEXT_PROMPT` | Given the scene graph (objects, attributes, relations, bboxes), produce 5 diverse first-person scenarios. Each scenario contains a `background`, a `goal`, and `relevant_objects`. Must be grounded in image content but not reveal fine-grained visual details. |

### Phase 2 — Multi-Round Conversation (GPT-4o)

| Prompt | Role |
|--------|------|
| `CONV_SYSTEM_PROMPT` | System prompt establishing the examiner role: simulate a natural conversation to probe VLM hallucinations. Embeds image info + context. Rules: multi-turn, human-like tone, no metadata leakage, prefer open-ended questions. |
| `SWITCH_PROMPT_EARLY` / `SWITCH_PROMPT_LATE` | Decision prompt — choose the next question type. Early (rounds < 6): 4 types. Late (rounds >= 6): adds type 5 "End conversation". Tracks previously asked types for diversity. |
| `REGULAR_CONV_PROMPT` | Generate a regular question about a specific target object node, grounded in context. De-duplicates against prior questions. |
| `FOLLOW_UP_CONV_PROMPT` | Generate a follow-up question to interrogate / challenge the VLM's last answer — test confidence, probe details, check consistency. |
| `ADVERSARIAL_CONV_PROMPT1` | Generate an adversarial question about a plausible-but-absent object (GT is always "No"). Targets things that commonly co-occur with visible objects but aren't in the image. |
| `UNANSWERABLE_CONV_PROMPT1` / `2` / `3` | 3-step chain-of-thought: (1) hallucinate a plausible absent object, (2) fabricate a relation to a real object, (3) produce a presuppositional trap question that assumes the hallucinated entity exists. |

### Question Types Summary

| ID | Type | Description |
|----|------|-------------|
| 1 | Regular | Directly about objects/attributes in the image, tied to the context. |
| 2 | Follow-up | Interrogates the VLM's last answer — challenges confidence, probes details, tests consistency. |
| 3 | Adversarial | Asks about plausible but absent objects; GT is always "No". |
| 4 | Unanswerable | Presuppositional trap — assumes a hallucinated object exists and asks for a specific detail about it. |
| 5 | End | Terminate the conversation (only available after round 5; requires double confirmation). |

## Conversation Loop Logic

- **Round 0**: always a regular question (type 1).
- **Rounds 1–5**: GPT-4o picks from types 1–4.
- **Rounds 6+**: type 5 (end) becomes available; requires double-confirmation to terminate.
- **Hard cap**: `--max_rounds 20` (default).
- Each turn: GPT-4o generates question + ground truth, question is sent to the VLM, VLM response is appended to conversation history.

## GPT Call Audit — What Each Call Receives

`LLMChat` (`utils/llm.py`) is **text-only** — it calls `client.chat.completions.create` with plain text messages. **No actual image is ever sent to GPT.** The parsed VG scene graph (`image_info`) is the text proxy for the image. The only place a real image is used is the `eval_func` call to the target VLM.

### GPT-5 calls (`llm_chat_context`) — context generation

| # | Method | Line | VG in prompt? | Image sent? | What's in the prompt |
|---|--------|------|:---:|:---:|---|
| 1 | `generate_context` | 562/572 | **YES** | No | `CONTEXT_PROMPT.format(image_info)` — full parsed VG scene graph injected into user message |

### GPT-4o calls (`llm_chat_conv`) — conversation driving

Every GPT-4o call passes a `conversations` list that **always starts with** `CONV_SYSTEM_PROMPT.format(self.image_info, ...)` as the system message. Therefore the full VG scene graph is in the system message of **every single GPT-4o call**. The conversation history also accumulates on each turn.

| # | Method | Line | VG in system msg? | VG in user msg? | What the user-turn adds |
|---|--------|------|:---:|:---:|---|
| 2 | `switch` | 481 | **YES** | No | `SWITCH_PROMPT_EARLY/LATE` — question-type selection + history of types |
| 3 | `ask_regular` | 492 | **YES** | No | `REGULAR_CONV_PROMPT` — single target node string + context |
| 4 | `ask_follow_up` | 500 | **YES** | No | `FOLLOW_UP_CONV_PROMPT` — short, no VG |
| 5 | `ask_adversarial` | 531 | **YES** | No | `ADVERSARIAL_CONV_PROMPT1` — context only, no VG |
| 6 | `ask_unanswerable` step 1 | 508 | **YES** | No | `UNANSWERABLE_CONV_PROMPT1` — context only, no VG |
| 7 | `ask_unanswerable` step 2 | 512 | **YES** | No | `UNANSWERABLE_CONV_PROMPT2` — short follow-up |
| 8 | `ask_unanswerable` step 3 | 517 | **YES** | No | `UNANSWERABLE_CONV_PROMPT3` — short follow-up |

### VLM call (not GPT)

| # | Method | Line | VG sent? | Image sent? | Notes |
|---|--------|------|:---:|:---:|---|
| 9 | `eval_func` | 753 | No | **YES** | Actual image file + question text sent to the target VLM being evaluated |

### Cost Implications

The parsed VG scene graph (`self.image_info`) is embedded in:

1. **`CONTEXT_PROMPT`** — sent once per sample to GPT-5.
2. **`CONV_SYSTEM_PROMPT`** — the system message for **every single GPT-4o call**.

Per context (1 of 5 per sample), a typical conversation involves:
- ~10–20 rounds of questions
- Each round: 1 switch call + 1 question-generation call (unanswerable = 3 calls)
- Roughly **20–40 GPT-4o calls per context**, each carrying the full VG string in system + growing conversation history

Across 5 contexts × 100 samples, the VG scene graph gets transmitted roughly **10,000–20,000 times** total. Since VG scene graphs can be very long (thousands of tokens describing all objects, attributes, and relations), this is the dominant cost driver.

---

## Diff Between `dyna_conv_v18resume.py` and `dyna_conv_v18resume5.py`

Only the `CONTEXT_PROMPT` differs (2 lines):

| Aspect | `v18resume` | `v18resume5` |
|--------|-------------|--------------|
| Diversity phrasing | "The **two** contexts must be meaningfully different" | "The **five** contexts must be meaningfully different" |
| Explicit diversity definition | (none) | Adds: *"Diversity is defined as the context/goal being able to provoke exploration of DIFFERENT objects / attributes of objects / relations between objects within the image."* |
| Instruction 3 | "generate several different and diverse contexts" | "generate several different and diverse contexts **and goals**" |

`resume5` strengthens the diversity constraint so that contexts explore genuinely different parts of the scene graph rather than rephrasing similar scenarios.

## Shell Script Diff (`final_run_all_parallel_full.sh` vs `final_run_all_parallel_full_resume5.sh`)

| Aspect | `full` | `resume5` |
|--------|--------|-----------|
| `RUN_FILE` | `examiner/dyna_conv_v18resume.py` | `examiner/dyna_conv_v18resume5.py` |
| `SAVE_DIR` | `work_dirs/vg/final_run_v18_gpt4o` | `work_dirs/vg/final_run_v18_gpt4o_resume5` |
| `LOG_DIR` | `work_dirs/logs_parallel` | `work_dirs/logs_parallel_resume5` |

**Models only in `full`:** BLIP2 (flan-t5-xl, flan-t5-xxl), InstructBLIP (vicuna-7b, vicuna-13b, flan-t5-xxl), Gemma 3 (4b, 12b, 27b), Phi-3.5-vision, Phi-4-multimodal, PaliGemma2 (10b, 28b), Ovis2 (1B, 2B, 34B), LLaVA-RLHF-7b, idefics2-8b-lpoi, InternVL3-38B.

**Models only in `resume5`:** opera-llava-1.5 (uses the older `dyna_conv_v18.py`, dedicated GPU 0).

**Shared models:** Qwen2.5-VL (3B, 7B, 72B), Qwen3-VL (2B, 8B, 32B), LLaVA-1.5 (7b, 13b), InternVL2 (2B, 8B, 26B), InternVL2.5 (2B, 8B, 38B), InternVL3 (2B, 8B).
