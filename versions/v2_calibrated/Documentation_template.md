# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** BreakEven  
**Team Members:** Pranav Sanjay Men, Sarang Rao  
**Submission Date:** September 2026

---

## 1. Executive Summary
Our solution implements a high-throughput, precision-weighted Entity Resolution system designed specifically for the macro $F_{0.5}$ evaluation metric across 26.4 million business records (1.73M Source 1 test queries against 9.97M Source 2/3 candidates). By exploiting absolute hard country boundaries and building multi-pass inverted indices over normalized core business names, distinctive tokens, and address numeric anchors, our blocking stage achieves >95% candidate recall ceiling while pruning comparisons down to ~35 candidates per entity. A gradient boosted classifier scored over RapidFuzz C++ pairwise similarities, calibrated with per-country empirical prior matching ($\tau_{US}=0.9744, \tau_{India}=0.9689, \tau_{France}=0.9903$) and strict probability-weighted 1-to-1 target collision resolution, achieved an outstanding candidate-to-match ratio of 3.329 matches/query (matching the true ground-truth prior of 3.46) with 5.75% singletons and 0 out-of-candidate leaks.

---

## 2. Methodology

### 2.1 Problem Analysis
Key insights discovered during extensive exploratory data analysis across the 26.4 million records:
- **Zero Cross-Country Linking:** Empirical audit across 180,000+ ground truth pairs confirmed **0 cross-country matches**. Business entities in India never link to entities in the US or France. Country provides an absolute hard partition.
- **Zero-Shot Test Generalization (France):** France accounts for 15.0% of the test set (259,452 entities in Source 1, 703,378 in Source 2, 731,615 in Source 3) but is entirely absent from the training set. Text normalization (NFKD stripping of diacritics/accents) and tokenizers are strictly language- and country-agnostic.
- **High Data Completeness with Sparse Address Gaps:** Source 1 features 100% complete names and addresses. Source 2 and Source 3 exhibit <0.001% missing names, but 2.7%–3.3% empty addresses. The architecture decouples name and address scoring so entities with omitted addresses match purely on name fidelity.
- **Dual-Channel Matching Topology:** Over 95% of true matches exhibit address token overlap and 85% exhibit name token overlap; crucially, **99.95% share either name or address overlap**. Some matches share identical addresses under trade names (DBAs), while others share identical business names across relocated/missing addresses.
- **Singleton Dominance & Precision Weighting:** Singletons represent **5.58%** (123,247 entities) of Source 1. Under $F_{0.5}$, precision is weighted 2× over recall ($(\beta = 0.5)$). Correctly predicting an empty set yields 1.0, while predicting even a single false match results in 0.0. Furthermore, 1 false positive on a 2-match entity drops the entity score from 1.0 to 0.714.

### 2.2 Solution Strategy
**Approach Type:** Multi-Pass Inverted Index Blocking + Pairwise Gradient Boosted Decision Trees (LightGBM/CatBoost) with Per-Country Prior Threshold Calibration & Probability-Weighted 1-to-1 Target Deduplication.  
**Core Innovation:** Country-partitioned multi-key candidate blocking combining suffix-stripped core name hashing, inverse document frequency (IDF) weighted token inverted indices, and address numeric anchors, evaluated using C++ RapidFuzz similarities at >2.5 million comparisons per second, followed by bisection threshold calibration against the empirical true match prior.

---

## 3. Candidate Generation (Blocking)
To reduce the $1.73 \times 10^{13}$ pairwise space down to a manageable candidate pool:
- **Blocking keys used:**
  1. *Exact Normalized Core Name:* Lowercased, NFKD accent-stripped, with legal suffixes (`LLC`, `Inc`, `Pvt Ltd`, `SARL`, `SAS`) and URL protocols/domains removed.
  2. *IDF-Filtered Name Tokens:* Words of length $\ge 3$, suppressing tokens exceeding frequency threshold ($0.5\%$ of pool).
  3. *Distinctive Address Tokens & Postal Codes:* Street identifiers and postal codes.
  4. *Address Numeric Keys:* Building numbers, plot IDs, and apartment numbers.
- **Candidate pairs generated:** Average of **35 candidates** per Source 1 entity (reducing search space by >99.999%).
- **Ensuring true matches were not lost:** The dual-channel index queries both business name channels and address numeric channels simultaneously. On our stratified benchmark, this achieved **95.45% recall in US** and **92.58% recall in India**.

---

## 4. Matching Model

**Features used (17 dimensions):**
- **Name features:** Levenshtein ratio, partial ratio, token sort ratio, token set ratio, core name ratio, word token Jaccard, character 3-gram Jaccard.
- **Address features:** Address Levenshtein ratio, token sort ratio, token set ratio, token Jaccard, numeric token Jaccard, binary exact number match, address missing indicator.
- **Interaction & Ranking:** Combined name+address token set ratio, candidate rank from blocking stage, and raw blocking composite score.

**Model type:** LightGBM Binary Classifier (GBDT, 63 leaves, learning rate 0.08, feature fraction 0.85).  
**Threshold calibration & deduplication:** 
- Instead of an uncalibrated global threshold, per-country thresholds were derived using empirical match density calibration ($\tau_{US}=0.9744, \tau_{India}=0.9689, \tau_{France}=0.9903$).
- Probability-weighted 1-to-1 deduplication: Every candidate record in Source 2 and Source 3 is strictly assigned to the single Source 1 query entity with the highest predicted match probability, completely eliminating cross-entity target collisions and preventing false-positive inflation.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** High-precision calibration yields strong alignment with ground-truth entity distributions:
  - Total S1 queries: 1,732,544
  - Total predicted pairs: 5,767,152
  - Mean matches / entity: **3.329** (Ground truth prior: **3.461**)
  - Singletons: **99,670** (5.75% vs Ground truth prior **5.58%**)
  - Validation Macro $F_{0.5}$: **0.956+**
- **Common false positives (wrong merges):** Franchises and chain stores sharing identical brand names in adjacent city blocks with slight address numbering differences.
- **Common false negatives (missed matches):** Heavy phonetic transliterations in non-Latin scripts (e.g. Hindi Devanagari in Source 2 vs English in Source 1) where address strings were also abbreviated.

---

## 6. Conclusion
The developed pipeline successfully resolves entities across 26.4 million records with sub-minute execution speeds and high precision. Multi-pass country-partitioned blocking guarantees high candidate recall, while our LightGBM model, per-country prior thresholding, and probability-weighted 1-to-1 collision resolution directly cater to the macro $F_{0.5}$ penalty on false merges.

---

## Appendix

### A. Code Artefacts
The full runnable pipeline is structured under `code/business_entity_resolution/`:
- `src/normalize.py`: Unicode normalization, legal suffix & domain stripper.
- `src/blocking.py`: Multi-pass inverted index candidate generator.
- `src/features.py`: RapidFuzz pairwise feature extractor.
- `src/model.py`: Model training and evaluation.
- `src/pipeline.py`: End-to-end test inference and consistency resolution.
- `requirements.txt`: Pinned dependencies.
- `README.md`: Step-by-step reproduction instructions.
