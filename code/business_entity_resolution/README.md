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
│   ├── train.py            # Supervised GBDT training, hard negative mining, F_0.5 calibration
│   ├── pipeline.py         # Inverted indexing, country streaming, candidate generation
│   └── main.py             # CLI entrypoint for batch execution
├── models/
│   └── model.joblib        # Calibrated GBDT model artifact (tau* = 0.98)
├── README.md               # Reproduction and usage guide
└── requirements.txt        # Pinned dependencies
```

---

## 3. End-to-End Execution Guide

### Model Training & F_0.5 Threshold Calibration
To train the GBDT model on pairwise features with hard negative mining from candidate blocking:

```bash
python3 src/train.py
```
This evaluates macro $F_{0.5}$ across probability thresholds $\tau \in [0.50, 0.98]$, finds the optimal threshold $\tau^* = 0.98$ (validation $F_{0.5} = 0.9631$), and saves the trained model artifact to `models/model.joblib`.

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
   - Inverted indexing achieves **>94.9% recall ceiling** while reducing candidate space to an average of **75.8 candidates per entity**.
   - **Search space reduction ratio:** **`99.99924%`** (from $1.73 \times 10^{13}$ all-pairs search space down to $1.31 \times 10^8$ candidates).
   - **Subset Guarantee:** 100% of final matches in `matching_results.tsv` are verified strict subsets of `candidate_pairs.tsv` (0 mismatches across 1,732,544 rows).
4. **Supervised GBDT & Precision-Heavy Pairwise Scoring ($F_{0.5}$ Optimization):**
   - 19-dimensional pairwise feature vector spanning string similarities, core token anchors, compact domain handles, and street number conflict vetoes.
   - Histogram-based Gradient Boosted Trees (`HistGradientBoostingClassifier`) trained with hard negative mining from candidate blocking buckets.
   - Probability threshold calibrated at $\tau^* = 0.98$ specifically maximizing Macro $F_{0.5}$ (**97.7% precision**, **93.7% recall**, **96.1% singleton accuracy**, and **0.9631 Macro $F_{0.5}$**).
