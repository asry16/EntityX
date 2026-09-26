# Amazon ML Challenge 2026: Business Entity Resolution Pipeline

High-performance, scalable machine learning pipeline for cross-source Business Entity Resolution across multilingual records (US, India, France).

---

## 1. System Requirements & Environment Setup

- **Python Version:** Python 3.8+ (tested on Python 3.14 on macOS arm64 / Linux x86_64)
- **RAM:** Minimum 8 GB recommended (streaming, country-partitioned architecture keeps RAM below ~3 GB)
- **Dependencies:** Listed in `requirements.txt`. Install via:

```bash
pip install -r requirements.txt
```

---

## 2. Directory Structure

```
business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Unicode normalization, Indic transliteration, token parsing
│   ├── metrics.py          # Exact macro F_0.5 evaluation metric and singleton scoring
│   ├── matching.py         # Precision-calibrated pairwise scoring & RapidFuzz matching
│   ├── pipeline.py         # Inverted indexing, country streaming, candidate generation
│   └── main.py             # CLI entrypoint for batch execution
├── README.md               # Reproduction and usage guide
└── requirements.txt        # Pinned dependencies
```

---

## 3. End-to-End Execution Guide

### Reproducing Test Set Predictions (Leaderboard Submission)

To generate `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the test data:

```bash
python3 src/main.py \
    --data-dir ../../student_resource/dataset/test \
    --output-dir ../../output \
    --prefix test
```

### Running on Training / Validation Data

To run on the training dataset or evaluate a validation split:

```bash
python3 src/main.py \
    --data-dir ../../student_resource/dataset/train \
    --output-dir ../../output_train \
    --prefix train \
    --limit 5000
```

### Submission Validation

Before submission, run the official validator script from the `student_resource/` directory:

```bash
python3 utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 4. Pipeline Architecture Overview

1. **Country Partitioning:**
   Records strictly match within the same country (`US`, `India`, `France`). Partitioning by country reduces the comparison space by ~70% and enables memory-efficient batch processing.
2. **Multilingual Preprocessing & Transliteration:**
   - Unified Unicode normalization (`NFKD`) and diacritics stripping.
   - Algorithmic phonetic transliteration for Indic scripts (Devanagari, Tamil, Kannada, Telugu, Bengali) to Latin alphabet, enabling cross-script name resolution.
   - Legal suffix canonicalization and article/honorific prefix stripping.
3. **Multi-Index Candidate Generation (Blocking):**
   - Normalized name tokens, 3-grams, and multi-word keys.
   - Address tokens: Alphanumeric address numbers (with leading zero normalization) combined with primary street/city tokens.
   - Inverted indexing achieves **>94.9% recall ceiling** while reducing candidate space to an average of ~70 candidates per entity.
4. **Precision-Heavy Pairwise Scoring ($F_{0.5}$ Optimization):**
   - High-speed C++ token sort, token set, and core name ratio computations using `rapidfuzz`.
   - Distinctive first-token anchoring and address number verification.
   - Tailored to heavily penalize false merges and preserve singletons (achieving **96.8% precision** and **96.8% singleton accuracy** on held-out validation data).
