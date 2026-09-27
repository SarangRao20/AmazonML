# Version 3 — Direct-Evidence Role-Decomposed Pipeline (Benchmark)

- **Submission ID:** Submission #3 (Submitted 27 Sep 2026, 03:54 PM IST)
- **Leaderboard Score ($F_{0.5}$):** **0.972865** (+0.1368 lift over V2)
- **Public Leaderboard Rank:** **1,275** / 6,868
- **Total Matches:** 5,718,652 (Mean: 3.301 matches / S1)
- **Singletons:** 101,181 (5.84%)
- **Target Collisions:** 0
- **Cross-Country Errors:** 0
- **Candidate Size:** 78,416,512 pairs (**45.26 candidates / S1**)
- **Matching TSV MD5:** `360f7e1f5efc5959e37ce276dfa25cd8`
- **Official Package:** `BreakEven_submission_0.9728.zip` (454 MB)

---

## 🔬 Methodology & Architecture
1. **Fine-Grained Address Role Decomposition:**
   - Instead of treating address strings as monolithic bags-of-words, parsed address tokens into granular semantic roles:
     - `premise_number` (house/plot/survey number)
     - `unit_number` (flat/suite/shop/office number)
     - `floor_number`
     - `street_name` / `landmark`
     - `city` / `postal_code` / `state`
2. **Feature Engineering (89 Features):**
   - High-precision lexical matching: Exact integer address equality, token set overlap, sorted ratio.
   - Indic script transliteration awareness (handling Devanagari, Bengali, Tamil, Kannada, Telugu, Punjabi, Gujarati script variations without falling back to blank matches).
   - Acronym extraction and token prefix matching.
3. **Model & Decision Architecture:**
   - LightGBM binary classifier trained with weighted log-loss penalizing false positives.
   - Inference with calibrated threshold $\tau = 0.80$.
   - Strict 1-to-1 exclusivity: $f: \{S2, S3\} \to S1 \cup \{\emptyset\}$. Exactly zero target collisions.
