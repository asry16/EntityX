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

### 3.3 Blocking Performance & Candidate Set Efficiency
- **Search Space Reduction Ratio:** **`99.99924%`** — reducing the theoretical $1.73\text{M} \times 9.97\text{M} \approx 1.73 \times 10^{13}$ all-pairs search space down to $1.31 \times 10^8$ candidates across the entire test set.
- **Candidate Set Compactness:** **75.82 candidates per Source 1 entity** on average across all 1,732,544 test entities.
- **Recall Ceiling:** **94.92%** of all ground-truth matches retained in candidate set.
- **Entities with 100% Matches Retained:** **85.82%**.
- **Candidate Subset Integrity:** Verified that **100%** of predicted matches in `output/matching_results.tsv` are strict subsets of `output/candidate_pairs.tsv` (0 mismatches across all 1,732,544 entities), adhering strictly to competition validation guidelines.

---

## 4. Matching Model & Supervised ML Architecture

Following the official AWS Builder Center Amazon ML Challenge 2026 guidelines, our matching engine employs a two-tier hybrid architecture combining high-speed candidate filtering with supervised Gradient Boosted Decision Tree (GBDT) pairwise classification.

### 4.1 Feature Engineering (19-Dimensional Pairwise Vector)
Pairwise features are evaluated via high-speed C++ vectorized routines (`rapidfuzz`):
1. **Rule Match Indicator:** Baseline precision-calibrated heuristic decision flag (1.0 / 0.0).
2. **String Similarity Metrics:** `name_token_sort_ratio`, `name_token_set_ratio`, `name_ratio`, `name_partial_ratio`.
3. **Core Token Anchoring:** `core_first_token_ratio` (Levenshtein ratio on first core token) and `has_first_match` boolean flag.
4. **Token Permutation & Transposition:** `core_sort_ratio` across all non-generic core tokens.
5. **Compact Alphanumeric Overlap:** `compact_sub` detecting handle/domain substring containment (e.g., `tmrindia` $\leftrightarrow$ `tmr (india)`).
6. **Physical Address Number Signals:**
   - `num_match`: Boolean set intersection on numeric digit stems (`get_num_stems`), handling unit letters (e.g., `31` $\leftrightarrow$ `31D`).
   - `num_conflict`: Absolute veto feature active when both records contain non-overlapping street numbers.
7. **Address String Overlaps:** `addr_token_sort_ratio`, `addr_token_set_ratio`, and `addr_partial_ratio`.
8. **Field Availability & Length Deltas:** `has_addr1`, `has_addr2`, `both_addr`, `abs(len(cn1) - len(cn2))`, and `abs(len(ca1) - len(ca2))`.

### 4.2 Supervised GBDT Training & Hard Negative Mining
- **Model Family:** Histogram-based Gradient Boosted Trees (`HistGradientBoostingClassifier`, scikit-learn's native C-accelerated LightGBM equivalent).
- **Hard Negative Mining:** Rather than training on trivial random negatives, negative pairs are mined directly from multi-key inverted index candidate buckets. This forces the tree to learn subtle discriminators between distinct businesses sharing identical street names, corporate suffixes, or postal codes.
- **Class Balancing:** Negatives per entity ratio calibrated at 6:1 to reflect the real-world candidate imbalance.

### 4.3 Macro $F_{0.5}$ Probability Threshold Calibration
Because the competition evaluates Macro $F_{0.5}$ with a steep penalty on false positive merges and singleton errors:
- Probability threshold $\tau^*$ is calibrated via grid search over $\tau \in [0.50, 0.98]$ on held-out validation data.
- The calibrated threshold assigns positive matches only when model confidence $P(y=1) \ge \tau^*$, maximizing precision and preserving true singletons.

---

## 5. Results & Error Analysis

### 5.1 Validation Performance
Evaluated under the official Macro $F_{0.5}$ metric against ground truth:

| Metric | Score |
| :--- | :--- |
| **Macro $F_{0.5}$ Score** | **0.9455 – 0.9584** |
| **Precision** | **96.90% – 98.30%** |
| **Recall** | **89.35% – 95.30%** |
| **Singleton Accuracy** | **96.77%** (High retention of true singletons) |
| **Candidate Blocking Recall** | **94.92%** |

### 5.2 Error Analysis
- **False Positives (Wrong Merges):** Primarily occurred when two distinct local entities shared identical common corporate suffixes (e.g., *Traders Private Limited*) and vague address terms without street numbers. Anchoring on the distinctive first token and explicit `num_conflict` veto eliminated 90%+ of these errors.
- **False Negatives (Missed Matches):** Occurred in rare cases where both the business name underwent an extreme DBA rename (e.g., *George Saul Inc* $\leftrightarrow$ *Umbrayuma*) AND the address was completely omitted or corrupt in the secondary source.

---

## 6. Conclusion

By exploiting strict country boundaries, native Indic script phonetic transliteration, multi-key inverted indexing, and GBDT pairwise classification calibrated for Macro $F_{0.5}$, our solution achieves state-of-the-art performance while maintaining a lightweight footprint that completes full inference across 1.73M entities in approximately 1 hour on commodity hardware without any external API lookups.

---

## Appendix

### A. Code Artefacts
The reproducible codebase is organized inside `code/business_entity_resolution/`:
- `src/preprocessing.py`: Multilingual text normalization, Indic transliteration, and key extraction.
- `src/metrics.py`: Exact macro-averaged $F_{0.5}$ score computation.
- `src/matching.py`: RapidFuzz pairwise scoring and decision logic.
- `src/train.py`: Supervised GBDT training, hard negative mining, and $F_{0.5}$ threshold calibration.
- `src/pipeline.py`: Inverted indexing, candidate streaming, and output formatting.
- `src/main.py`: CLI entry point (`python3 src/main.py --data-dir dataset/test --output-dir output`).
- `requirements.txt`: Pinned Python dependencies (`rapidfuzz`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `joblib`).
- `README.md`: Step-by-step reproduction instructions.
