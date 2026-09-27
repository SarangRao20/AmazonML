# Version 2 — Country Bisection Calibrated Pipeline

- **Submission ID:** Submission #2 (Submitted 27 Sep 2026, 01:47 PM IST)
- **Leaderboard Score ($F_{0.5}$):** **0.836000** (+0.1235 lift over V1)
- **Public Leaderboard Rank:** ~3,500
- **Total Matches:** 5,767,152 (Mean: 3.329 matches / S1)
- **Candidate Size:** 77,964,480 pairs (45.0 candidates / S1)

---

## 🔬 Methodology & Architecture
1. **Per-Country Target Match Count Calibration:**
   - Implemented binary bisection search on model probability threshold $\tau_c$ per country to force the mean match count per entity to align with the training ground truth prior (~3.46 matches / entity).
   - Calibrated thresholds: $\tau_{\text{US}} \approx 0.74$, $\tau_{\text{India}} \approx 0.76$, $\tau_{\text{France}} \approx 0.73$.
2. **Post-Processing:**
   - Applied global 1-to-1 target exclusivity (`tools/enforce_one_to_one.py`) to eliminate duplicate target assignments.

---

## 📉 Error Analysis & Key Finding
- **The France Zero-Shot Collapse:**
  - France has zero training records in `train_source*.tsv` and appears only in the test set.
  - Bisection threshold search was performed *before* 1-to-1 deduplication.
  - When global deduplication was applied, 134,000 target collisions were deleted.
  - Because France model scores had lower confidence margins (due to domain shift), French entities disproportionately lost matches to higher-confidence candidates.
  - The mean match count for France collapsed from 3.46 down to **2.942** (a 15% recall drop).
- **Lesson Learned:** Calibration and target conflict resolution must be coupled, and country-specific confidence distributions require separate post-hoc margins.
