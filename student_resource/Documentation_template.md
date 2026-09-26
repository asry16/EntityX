# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** FRIDAY  
**Team Members:** Asmita Roy, Ritu Raj  
**Submission Date:** 26th September 2026  

---

## 1. Executive Summary

We developed a high-throughput, multilingual entity resolution pipeline combining country-partitioned multi-key inverted index blocking with precision-calibrated pairwise scoring. Our solution features algorithmic Indic script phonetic transliteration (Devanagari, Tamil, Kannada, Telugu, Bengali) to resolve cross-script business entities without external API lookups. Evaluated under the competition's macro-averaged $F_{0.5}$ metric, our system achieved **94.92% candidate recall ceiling** during blocking and **0.9455 macro $F_{0.5}$** with **96.90% precision** and **96.77% singleton accuracy** on held-out validation data.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis across 2.2M+ training records revealed several distinct noise patterns and architectural opportunities:
1. **Zero Cross-Country Matches:** True matches strictly reside within the same country (`US`, `India`, `France`). No ground truth links span international borders.
2. **Multilingual Transliteration Gap:** In the Indian partition, Source 1 records feature standard English Latin script while Source 2/3 records frequently use native Indian scripts (e.g., *Raj Investments LLP* $\leftrightarrow$ *ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி* or *Ss Food* $\leftrightarrow$ *एसएस फूड प्राइवेट लिमिटेड*). In France (test only), accent variations (`é`, `è`, `ô`) and French legal abbreviations (`SARL`, `SASU`, `SCI`) predominate.
3. **Complementary Name/Address Signals:** Some records have identical business names with omitted/empty addresses, while others represent DBA/brand name changes sharing identical street numbers and local address tokens. A single-key blocking strategy bottlenecks recall; a multi-index union is mandatory.
4. **Severe Singleton Penalty:** ~5.58% of entities are singletons (0 matches). In the macro $F_{0.5}$ metric, predicting even a single spurious match drops an entity's score from 1.0 to 0.0. Conservative thresholding and distinctive token anchoring are critical.

### 2.2 Solution Strategy
**Approach Type:** Country-Partitioned Multi-Index Blocking + High-Precision Pairwise Matcher  
**Core Innovation:** Direct algorithmic Unicode phonetic transliteration for 5 major Indic scripts combined with core-token anchoring, ensuring that long generic corporate suffixes do not trigger false positive merges.

---

## 3. Candidate Generation (Blocking)

### 3.1 Partitioning & Inverted Indexing
To prevent the combinatorial explosion of $1.73\text{M} \times 10\text{M} \approx 1.7\times 10^{13}$ pairwise comparisons, candidate generation is partitioned by country:
1. **France:** ~260k S1 vs. ~1.43M S2/S3
2. **US:** ~663k S1 vs. ~3.82M S2/S3
3. **India:** ~810k S1 vs. ~4.72M S2/S3

### 3.2 Blocking Keys Used
- **Normalized Name Keys:**
  - `n1:<token>`: First core business name token (stripped of articles like *The*, *Mr*, *M/s*).
  - `np4:<prefix>` / `np3:<prefix>`: 3-gram and 4-gram prefixes of the primary token.
  - `n2:<token1>_<token2>`: Bigram of the first two core tokens.
  - `nt:<token>`: Secondary distinctive core tokens ($\ge 4$ characters).
- **Address Keys:**
  - `an:<num>_<street/city>`: Alphanumeric address numbers (with leading zero normalization, e.g., `af-0684` $\rightarrow$ `af684`, `0026` $\rightarrow$ `26`) paired with primary street or city words.
  - `zip:<postal>`: Standalone 5-digit US ZIP and 6-digit Indian PIN codes.

### 3.3 Blocking Performance (Validation Split)
- **Candidate pairs generated per entity:** Average 73.5 candidates.
- **Recall ceiling:** **94.92%** of all ground-truth matches retained in candidate set.
- **Entities with 100% matches retained:** **85.82%**.

---

## 4. Matching Model

### 4.1 Feature Engineering
Pairwise features are evaluated via high-speed C++ vectorized routines (`rapidfuzz`):
- `name_token_sort_ratio`: Token sort similarity across business names.
- `name_token_set_ratio`: Token set similarity handling sub-phrases and insertions.
- `core_first_token_ratio`: Levenshtein ratio on the distinctive non-generic first token.
- `has_first_match`: Boolean flag ensuring primary name anchor match ($\ge 65\%$ ratio or exact 1-2 character match).
- `addr_token_set_ratio`: Token set overlap on normalized street and city tokens.
- `addr_number_match`: Exact set intersection of normalized street/plot/unit numbers.

### 4.2 Precision-Calibrated Decision Rule
Given the precision-heavy nature of $F_{0.5}$ ($\beta = 0.5$, precision weighted $2\times$ over recall):
1. **Rule 1 (Name-Led Match):** `name_token_sort_ratio >= 85` AND `(has_first_match OR addr_number_match)`. If both records contain an address, `addr_token_set_ratio >= 45` or `addr_number_match == 1` is required.
2. **Rule 2 (Address-Led Match):** `(addr_token_set_ratio >= 70 OR (addr_number_match AND addr_token_set_ratio >= 50))` AND `(name_token_set_ratio >= 60 OR name_token_sort_ratio >= 55)` AND `(has_first_match OR addr_number_match)`.

---

## 5. Results & Error Analysis

### 5.1 Validation Performance
Measured on a representative 500-entity holdout set across 20,000+ distractor candidates:

| Metric | Score |
| :--- | :--- |
| **Macro $F_{0.5}$ Score** | **0.9455 (94.55%)** |
| **Precision** | **96.90%** |
| **Recall** | **89.35%** |
| **Singleton Accuracy** | **96.77%** (30 / 31 singletons correctly identified) |
| **Non-Singleton $F_{0.5}$** | **94.40%** |

### 5.2 Error Analysis
- **False Positives (Wrong Merges):** Primarily occurred when two distinct local entities shared identical common corporate suffixes (e.g., *Traders Private Limited*) and vague address terms without street numbers. Anchoring on the distinctive first token eliminated 90%+ of these errors.
- **False Negatives (Missed Matches):** Occurred in rare cases where both the business name underwent an extreme DBA rename (e.g., *George Saul Inc* $\leftrightarrow$ *Umbrayuma*) AND the address was completely omitted or corrupt in the secondary source.

---

## 6. Conclusion

By exploiting strict country boundaries, native Indic script phonetic transliteration, multi-key inverted indexing, and precision-anchored pairwise scoring, our solution achieves state-of-the-art $F_{0.5}$ performance while maintaining a lightweight footprint that completes full inference across 1.73M entities in approximately 1 hour on commodity hardware without any external API lookups.

---

## Appendix

### A. Code Artefacts
The reproducible codebase is organized inside `code/business_entity_resolution/`:
- `src/preprocessing.py`: Multilingual text normalization, Indic transliteration, and key extraction.
- `src/metrics.py`: Exact macro-averaged $F_{0.5}$ score computation.
- `src/matching.py`: RapidFuzz pairwise scoring and decision logic.
- `src/pipeline.py`: Inverted indexing, candidate streaming, and output formatting.
- `src/main.py`: CLI entry point (`python3 src/main.py --data-dir dataset/test --output-dir output`).
- `requirements.txt`: Pinned Python dependencies (`rapidfuzz`, `scikit-learn`, `pandas`, `numpy`, `scipy`).
- `README.md`: Step-by-step reproduction instructions.
