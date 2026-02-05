# CHAIR Metrics Explanation

CHAIR (Caption Hallucination Assessment with Image References) is a metric for evaluating object hallucination in vision-language models. This document explains each metric computed by the CHAIR evaluation system.

## Overview

CHAIR evaluates whether objects mentioned in model responses (captions/conversations) actually exist in the image by:
1. Extracting object nouns from the response
2. Matching them to WordNet synsets (semantic categories)
3. Comparing against ground truth objects from the image
4. Using WordNet hierarchy to allow semantic matches (e.g., "dog" matches "canine")

**Important**: Negative mentions (e.g., "there is no fork") are excluded from evaluation.

---

## Core Metrics

### 1. **CHAIRs** (CHAIR Sentence-level)
- **Definition**: Fraction of sentences/captions that contain at least one hallucinated object
- **Formula**: `num_hallucinated_caps / num_caps`
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Interpretation**: 
  - **Lower is better** (0.0 = no hallucinations, 1.0 = all sentences have hallucinations)
  - Measures how often the model produces sentences with false object mentions
  - Example: If 30 out of 100 sentences contain hallucinations, CHAIRs = 0.30 (30%)

### 2. **CHAIRi** (CHAIR Instance-level, original)
- **Definition**: Fraction of hallucinated object words out of all object words mentioned
- **Formula**: `hallucinated_word_count / vg_word_count`
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Interpretation**:
  - **Lower is better** (0.0 = no hallucinated words, 1.0 = all words are hallucinations)
  - Measures the density of hallucinations in the text
  - Example: If 15 out of 200 object words are hallucinations, CHAIRi = 0.075 (7.5%)
  - **Note**: Counts each word occurrence separately (may count same word multiple times)

### 3. **CHAIRi_v2** (CHAIR Instance-level, deduplicated)
- **Definition**: Fraction of unique hallucinated objects out of all unique objects mentioned
- **Formula**: `len(unique_hallucinated_words) / len(unique_all_words)`
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Interpretation**:
  - **Lower is better** (0.0 = no unique hallucinations, 1.0 = all unique words are hallucinations)
  - Similar to CHAIRi but deduplicates repeated mentions
  - More robust to models that repeat the same hallucinated word multiple times
  - Example: If 10 unique hallucinated objects out of 150 unique objects, CHAIRi_v2 = 0.067 (6.7%)

---

## Coverage Metrics

### 4. **Coverage_avg** (Average Coverage per Image)
- **Definition**: Average fraction of ground truth objects that were correctly mentioned across all images
- **Formula**: `mean(covered_objects_per_image / gt_objects_per_image)` for each image
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Interpretation**:
  - **Higher is better** (1.0 = all GT objects mentioned, 0.0 = no GT objects mentioned)
  - Measures how well the model covers the objects present in images
  - Computed per-image then averaged, so each image contributes equally
  - Example: Image 1: 8/10 objects covered (0.8), Image 2: 5/10 objects covered (0.5) → Coverage_avg = 0.65

### 5. **Coverage_all** (Overall Coverage)
- **Definition**: Fraction of all unique ground truth objects (across all images) that were mentioned
- **Formula**: `len(unique_mentioned_gt_objects) / len(unique_all_gt_objects)`
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Interpretation**:
  - **Higher is better** (1.0 = all GT objects mentioned at least once, 0.0 = no GT objects mentioned)
  - Measures overall recall of ground truth objects
  - Aggregates across all images (not per-image average)
  - Example: If 500 unique GT objects exist and 350 were mentioned, Coverage_all = 0.70 (70%)

---

## Per Question Type Metrics

When using `--by_qtype`, the same metrics are computed separately for each question type:

- **regular**: Direct questions about the image
- **follow-up**: Questions following up on previous responses
- **adversarial**: Questions about plausible but absent objects
- **unanswerable**: Questions with false presuppositions

Each question type reports:
- `{type}_CHAIRs`: Sentence-level hallucination rate for this type
- `{type}_CHAIRi`: Instance-level hallucination rate for this type
- `{type}_CHAIRi_v2`: Deduplicated instance-level rate for this type
- `{type}_Coverage_avg`: Average coverage for this type
- `{type}_Coverage_all`: Overall coverage for this type
- `{type}_num_caps`: Number of sentences of this type
- `{type}_num_hallucinated_caps`: Number of sentences with hallucinations of this type

---

## Key Differences

| Metric | What it measures | Aggregation |
|--------|------------------|-------------|
| **CHAIRs** | How many sentences have hallucinations | Sentence-level |
| **CHAIRi** | How many words are hallucinations | Word-level (with duplicates) |
| **CHAIRi_v2** | How many unique objects are hallucinations | Word-level (deduplicated) |
| **Coverage_avg** | How well GT objects are covered | Per-image average |
| **Coverage_all** | Overall GT object recall | Global aggregation |

---

## Example Interpretation

For a model with:
- **CHAIRs = 0.25** (25%): 1 in 4 sentences contains a hallucination
- **CHAIRi = 0.10** (10%): 1 in 10 object words is a hallucination
- **CHAIRi_v2 = 0.08** (8%): 1 in 12.5 unique objects is a hallucination
- **Coverage_avg = 0.70** (70%): On average, 70% of GT objects per image are mentioned
- **Coverage_all = 0.65** (65%): 65% of all unique GT objects were mentioned at least once

This model:
- ✅ Has moderate hallucination rates (10-25%)
- ✅ Has good coverage (65-70% of objects mentioned)
- ⚠️ Still produces some false object mentions

---

## Notes

1. **WordNet Hierarchy**: CHAIR uses WordNet to allow semantic matches. For example, if the image contains a "dog" and the model says "canine", this is considered correct due to hypernym/hyponym relationships.

2. **Negative Mentions**: Phrases like "there is no fork" or "without a knife" are excluded from evaluation, as they correctly state the absence of objects.

3. **Physical Objects Only**: Only physical objects (nouns) are evaluated, not attributes or relationships directly.

4. **Stemming**: Words are stemmed before matching (e.g., "dogs" → "dog") to handle pluralization.
