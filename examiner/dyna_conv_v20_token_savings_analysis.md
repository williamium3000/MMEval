# v20 Examiner Token Savings vs v18

This note estimates how many **examiner-side LLM input tokens** `dyna_conv_v20.py` saves relative to:

- `dyna_conv_v18resume5.py`
- `dyna_conv_v19.py`

Assumption:
- Compare against `v18resume5`, since that is the 5-context `v18` flow documented in [dyna_conv_v18_flow_analysis.md](/raid/william/project/context-eval-mllm/examiner/dyna_conv_v18_flow_analysis.md).
- Use the completed conversations in `work_dirs/vg/final_run_v20_gpt4o_small/InternVL2_5-8B_cache.json` as the concrete sample.
- Reconstruct examiner prompt load from saved conversation traces.
- Token estimate uses the same rough estimator as `utils/llm.py`: `chars // 4 + 1`.

## What changed from v18 to v20

`v20` reduces examiner tokens through 4 main mechanisms:

1. **5 contexts -> 2 contexts**
   - `v18resume5` generates 5 contexts per image.
   - `v20` generates 2 contexts per image.

2. **Full VG -> compact VG in phase 1**
   - `v18resume5` uses `format_case_vg`.
   - `v20` uses `format_case_vg_compact`.

3. **Full VG -> per-context subset VG in phase 2**
   - `v18resume5` injects the full scene graph into every conversation-driving GPT call.
   - `v20` injects only the phase-1.5 subset: selected relevant objects + adjacent relational neighbors.

4. **3-call unanswerable chain -> 1 structured call**
   - `v18resume5` uses `UNANSWERABLE_CONV_PROMPT1/2/3`.
   - `v20` uses one structured `UNANSWERABLE_CONV_PROMPT`.

## Sample used

Concrete sample:
- File: `work_dirs/vg/final_run_v20_gpt4o_small/InternVL2_5-8B_cache.json`
- Conversations analyzed: `10`
- Average rounds per conversation: `13.2`
- Question type counts across these 10 conversations:
  - `regular`: `46`
  - `follow-up`: `29`
  - `adversarial`: `43`
  - `unanswerable`: `14`

## Scene Graph Size Reduction

Average estimated prompt tokens for the scene graph string on these 10 cached conversations:

| Representation | Avg tokens |
|---|---:|
| `v18` full VG | `1033.2` |
| `v20` compact VG | `423.0` |
| `v20` phase-1.5 subset VG | `314.2` |

Observed reductions:
- Full VG -> compact VG: **59.06% smaller**
- Full VG -> subset VG: **69.59% smaller**
- Compact VG -> subset VG: **25.72% smaller**

This matters because in `v18`, the full VG is repeated inside the **system prompt of every GPT-4o conversation-driving call**. In `v20`, that repeated payload is much smaller.

## Estimated Examiner Tokens Per Conversation

Using the same saved conversations, but scoring them under the two different prompt architectures:

| Architecture | Avg examiner input tokens per conversation |
|---|---:|
| `v18resume5`-style | `70,966.1` |
| `v20` | `47,401.6` |

Savings on the **same conversation trace**:
- **23,564.5 tokens saved per conversation**
- **33.21% reduction**

Interpretation:
- This isolates the prompt-architecture effect.
- It includes compact/subset SG usage and the 1-call unanswerable flow.
- It does **not** yet include the 5-context -> 2-context reduction.

## Estimated Examiner Tokens Per Image

Per-image estimate:
- `v18resume5`: `1` context-generation call + `5` conversations
- `v20`: `1` context-generation call + `2` conversations

Average context-generation call cost from the same samples:

| Architecture | Avg context-generation input tokens |
|---|---:|
| `v18resume5` | `2,594.0` |
| `v20` | `1,722.6` |

Estimated total examiner input tokens per image:

| Architecture | Avg tokens per image |
|---|---:|
| `v18resume5` | `357,424.5` |
| `v20` | `96,525.8` |

Overall estimated savings per image:
- **260,898.7 tokens saved per image**
- **72.99% reduction**

## Comparison With v19

`v19` already includes two major savings that `v20` inherits:

1. `format_case_vg_compact` in phase 1
2. `2` contexts per image instead of `5`

So the incremental `v20` vs `v19` savings come mainly from:

1. **phase-1.5 subset SG in phase 2**
   - `v19`: phase 2 still injects the full **compact** VG into every conversation-driving call
   - `v20`: phase 2 injects only the **per-context subset** VG

2. **1-call unanswerable generation**
   - `v19`: 3 calls
   - `v20`: 1 structured call

### Same Conversation Trace: v19 vs v20

Using the same 10 cached conversations:

| Architecture | Avg context-generation input tokens | Avg conversation input tokens | Avg total input tokens |
|---|---:|---:|---:|
| `v19` | `1,722.6` | `54,425.3` | `56,147.9` |
| `v20` | `1,722.6` | `47,401.6` | `49,124.2` |

Savings on the **same 2-context architecture / same conversation trace**:

- Conversation-only savings: **7,023.7 tokens**
- Conversation-only reduction: **12.91%**
- Total reduction per reconstructed context item: **12.51%**

Important detail:
- `v19` and `v20` have the **same phase-1 context-generation cost** in this comparison.
- That is why `v20` vs `v19` context-generation savings are **0%**.
- The gain comes almost entirely from **phase-2 prompt reduction**.

### Per-Image Estimate: v19 vs v20

Because both use:
- `1` context-generation call per image
- `2` conversations per image

the per-image comparison is straightforward:

| Architecture | Avg tokens per image |
|---|---:|
| `v19` | `110,573.2` |
| `v20` | `96,525.8` |

Per-image `v20` savings vs `v19`:
- **14,047.4 tokens saved per image**
- **12.70% reduction**

## Takeaway

If we compare `v20` against `v18resume5`:

- **Per conversation**, `v20` saves about **33%** of examiner input tokens.
- **Per image**, once the **5 contexts -> 2 contexts** change is included, `v20` saves about **73%** of examiner input tokens.

So the short answer is:

> `v20` saves roughly **one-third** of examiner tokens on the same conversation flow, and roughly **three-quarters** of examiner tokens per image relative to `v18resume5`.

If we compare `v20` against `v19`:

> `v20` saves about **12.9%** of examiner conversation tokens, and about **12.7%** of examiner tokens per image.

## Why the per-image savings are so large

The largest driver is still **repetition of scene-graph text across many GPT calls**.

`v20` attacks that from three directions at once:

1. fewer contexts per image
2. smaller VG string in phase 1
3. smaller per-context subset VG in phase 2

The 1-call unanswerable change helps too, but it is a secondary gain relative to the scene-graph repetition savings.

## Limitations

This is an **estimate**, not billing data.

Reasons:
- Token count is approximated by `chars // 4 + 1`.
- The reconstruction uses saved conversation traces from a `v20` run.
- The comparison asks: “what would these same conversations cost under `v18resume5` prompts?”
- Real `v18` runs may diverge slightly because the larger prompt can itself change GPT behavior and conversation length.

Despite that, the direction and scale are stable:
- modest-to-large savings per conversation
- very large savings per image
