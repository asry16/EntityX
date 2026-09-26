# EntityX: Business Entity Resolution Pipeline

**Team:** FRIDAY  
**Members:** Asmita Roy, Ritu Raj  
**Competition:** Amazon ML Challenge 2026  

---

## Overview

EntityX is an end-to-end, high-throughput machine learning solution designed to resolve business identity records across multiple independent commercial data sources with noisy, incomplete, and multilingual fields (covering US, India, and France).

---

## Key Innovations

1. **Country Partitioning:** Segregates comparisons by country (`US`, `India`, `France`), reducing the $1.73\text{M} \times 10\text{M} \approx 1.7\times 10^{13}$ pairwise space by ~70% and ensuring strict bounded memory consumption (< 3GB RAM).
2. **Algorithmic Indic Transliteration:** Zero-external-lookup Unicode transliteration supporting Devanagari, Tamil, Kannada, Telugu, and Bengali to Latin script.
3. **Multi-Index Blocking:** Combines normalized business name tokens, prefixes (3-grams, 4-grams), numeric stems, and address components, yielding a **>95% candidate recall ceiling**.
4. **Precision-Calibrated Pairwise Matcher:** Rapid vectorized string comparisons via `rapidfuzz` (Token Sort, Token Set, Core First-Token Ratio) with street number conflict rejection, tuned specifically for the macro-averaged $F_{0.5}$ metric.

---

## Repository Structure

```
├── code/
│   └── business_entity_resolution/
│       ├── requirements.txt            # Pinned dependencies
│       ├── README.md                   # Reproduction instructions
│       └── src/
│           ├── __init__.py
│           ├── preprocessing.py        # Text cleaning & Indic transliteration
│           ├── metrics.py              # Macro F_0.5 evaluation metric
│           ├── matching.py             # Pairwise feature scoring rules
│           ├── pipeline.py             # Inverted indexing & streaming engine
│           └── main.py                 # CLI execution entry point
├── student_resource/
│   ├── utils/
│   │   └── validate_submission.py      # Official submission validator
│   └── Documentation_template.md       # Solution write-up
├── Documentation_template.md           # Methodology write-up
└── .gitignore                          # Data & artifact exclusions
```

---

## Reproduction & Usage

### 1. Install Dependencies
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Run Inference Pipeline
```bash
python3 code/business_entity_resolution/src/main.py \
    --data-dir student_resource/dataset/test \
    --output-dir output \
    --prefix test
```

### 3. Validate Submission Outputs
```bash
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
