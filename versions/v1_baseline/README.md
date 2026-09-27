# Version 1 — Baseline Relative Pipeline

- **Submission ID:** Submission #1 (Submitted 27 Sep 2026, 12:32 PM IST)
- **Leaderboard Score ($F_{0.5}$):** **0.712498**
- **Public Leaderboard Rank:** ~4,200
- **Total Matches:** 7,466,211 (Mean: 4.309 matches / S1)
- **Candidate Size:** 77,964,480 pairs (45.0 candidates / S1)

---

## 🔬 Methodology & Architecture
1. **Candidate Generation (Blocking):**
   - Inverted token indexing on normalized business names and address components.
   - Constrained candidate search to top 45 candidates per S1 entity within each country.
2. **Matching Model:**
   - LightGBM binary classifier trained on pairwise lexical & phonetic features (Levenshtein, token sort, Jaro-Winkler, Soundex).
3. **Decision Rule:**
   - Relative thresholding: accepted matches with predicted probability $\ge 0.65$.
   - Greedy local deduplication.

---

## 📉 Error Analysis & Why It Scored 0.712
- **Over-prediction of Matches:** Mean matches reached 4.309 per S1 entity, significantly exceeding the true training prior (~3.46–3.66).
- **Asymmetric Metric Penalty:** Macro $F_{0.5} = \frac{5 \cdot TP}{T + 4 \cdot P}$ penalizes False Positives $4\times$ harder than missed matches. Generating ~1.7 million excess false positive matches heavily dragged down precision and bounded the score to ~0.71.
- **Root Cause Identified:** The threshold $\tau=0.65$ was uncalibrated on the 10-million entity test pool where negative candidate density is vastly higher than small validation splits.
