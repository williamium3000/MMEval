# Token Analysis for Examiner v18

## Cost Breakdown (100 samples × 5 contexts × ~10 rounds)

| Component | Per Context | Full Run | Share |
|-----------|------------|----------|-------|
| **VG scene graph** | ~24,800 tok | **12.5M** | **43%** |
| **CONV_SYSTEM_PROMPT static** | ~12,400 tok | **6.2M** | **22%** |
| **Instruction prompts** | ~10,100 tok | **5.1M** | **18%** |
| **Conversation history** | ~9,900 tok | **5.0M** | **17%** |
| **Total** | **~57,200 tok** | **~28.8M** | **100%** |

### What IS in conversation history (grows ~90 tok/round)

Only the examiner's question (sent to the VLM) and the VLM's response persist in the `conversations` list between rounds (lines 751–755). GPT-4o's own instruction prompts and raw JSON outputs are **not** kept.

```
messages[0] = system (VG + CONV_SYSTEM_PROMPT)                          ← counted as VG/static above
messages[1] = assistant: "Can you spot the parking meter?"               ~21 tok  ← question asked to VLM
messages[2] = user: "the parking meter is orange, on the right side..." ~67 tok  ← VLM response
messages[3] = assistant: "You mentioned no orange meter, but could..."   ~21 tok  ← question asked to VLM
messages[4] = user: "upon reviewing again, I do not see..."             ~67 tok  ← VLM response
...
```

Real VLM response lengths (measured from output files):

| Model | Avg response | Avg question | Per-round growth | History at R=10 |
|-------|-------------|-------------|-----------------|-----------------|
| Qwen2.5-VL-7B | 96 tok (385 chars) | 21 tok | ~118 tok | ~1,175 tok |
| LLaVA-1.5-7b | 46 tok (183 chars) | 21 tok | ~66 tok | ~663 tok |
| InternVL3-8B | 59 tok (237 chars) | 21 tok | ~80 tok | ~802 tok |

### What is NOT in conversation history (discarded after each call)

All instruction prompts are appended to a `deepcopy()` of conversations and never persist:

| Prompt | Tokens | Sent per... |
|--------|--------|-------------|
| `SWITCH_PROMPT_EARLY/LATE` | 472–542 | every round |
| `REGULAR_CONV_PROMPT` | 275 | regular rounds |
| `FOLLOW_UP_CONV_PROMPT` | 407 | follow-up rounds |
| `ADVERSARIAL_CONV_PROMPT1` | 353 | adversarial rounds |
| `UNANSWERABLE_CONV_PROMPT1/2/3` | 488+52+597 | unanswerable rounds (3 calls) |

These are the "instruction prompts" category — ~18% of total cost. They are large but each is used once per round and discarded.

### Cost reduction levers

| Lever | Savings |
|-------|---------|
| Shrink VG format (compact → ~50% of current) | ~6M tokens (21%) |
| Reduce 5 → 2 contexts per sample | ~17M tokens (60%) |
| Both combined | ~26M → ~6M (77%) |

---

## Parsed VG Format (`format_case_vg` in `utils/vg.py`)

Two sections, no images sent — text-only proxy:

**Section 1 — Instances** (one line per object):
```
instance {id}, {name}, bbox: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}), attributes: {attr1, attr2, ...}
```

**Section 2 — Relations** (one line per relationship):
```
{subject_name} (instance {subject_id}) {predicate} {object_name} (instance {object_id})
```

Region descriptions exist in the data but are **not included** (`use_region=False`).

## Token Estimates (5 samples)

| Sample | image_id | Objects | Relations | Chars | ~Tokens |
|--------|----------|---------|-----------|-------|---------|
| 0 | 1 | 40 | 41 | 4,830 | 1,207 |
| 1 | 2 | 27 | 23 | 3,055 | 763 |
| 2 | 3 | 52 | 50 | 6,292 | 1,573 |
| 3 | 4 | 24 | 16 | 2,633 | 658 |
| 4 | 5 | 34 | 28 | 3,842 | 960 |

- **Average: ~1,032 tokens** per VG string
- Range: 658–1,573
- Per instance line: ~72 chars avg
- Per relation line: ~46 chars avg

## Where VG Is Sent

The full VG string is embedded in **every GPT call** (see `dyna_conv_v18_analysis.md`). It is never filtered by `relevant_objects` — those only guide which object node `ask_regular` targets next. A typical run of 100 samples × 5 contexts × ~30 GPT-4o calls/context means the VG is transmitted **~15,000 times**, totaling roughly **15M input tokens** just for VG alone.

## Redundancy in Current Format

| Issue | Example | Waste |
|-------|---------|-------|
| Repeated `instance` prefix | `instance 8, teddy bear, bbox: ...` | ~12 chars/line |
| Verbose bbox | `bbox: (0.56, 0.00, 1.00, 0.63)` = 33 chars | Could be `[.56,.00,1.0,.63]` (18 chars) or 1-decimal `[.6,.0,1,.6]` (12 chars) |
| `attributes: none` | Printed even when empty | ~18 chars/line wasted |
| Duplicate relations | `teddy bear against pillow` appears twice in sample 3 | Exact duplicates |
| Noisy objects | `color` / `colour` instances that are just attribute fragments | Whole lines wasted |
| Name repetition in relations | `teddy bear (instance 8) against pillow (instance 3)` = 55 chars | Names already in Instances section; `8 against 3` = 7 chars |

## Compact Format Suggestion

Current (sample 3, 24 objects, 16 rels → 2,633 chars / ~658 tokens):
```
Instances:
instance 8, teddy bear, bbox: (0.33, 0.52, 0.41, 0.64), attributes: stuffed, forward
...
Relation between the above instances:
teddy bear (instance 8) against pillow (instance 3)
```

Compact alternative:
```
Objects:
8: teddy bear [.33,.52,.41,.64] stuffed, forward
...
Relations:
8 against 3
```

**Estimated shrinkage:**
- Compact format alone (keep 2-decimal bbox): **~50%** of current size
- Plus 1-decimal bbox + drop noisy objects + dedup relations: **~30%** of current size
- On a 1,000-token VG, that's **300–500 tokens** — saving **500–700 tokens per GPT call**

