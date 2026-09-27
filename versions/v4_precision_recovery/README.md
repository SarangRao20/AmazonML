# Version 4 — Precision Recovery Pipeline

- **Submission ID:** Submission #4 (Ready / Submitted for Testing)
- **Projected Score ($F_{0.5}$):** **~0.975 – 0.977**
- **Total Matches:** 5,774,752 (Mean: 3.333 matches / S1)
- **Singletons:** 98,912 (5.71%)
- **Target Collisions:** 0
- **Cross-Country Errors:** 0
- **Candidate Size:** 78,416,512 pairs (45.26 candidates / S1)
- **Matching TSV MD5:** `3433d8bb5512b8290744fe6171a41dbe`
- **Output Directory:** `output_enhanced_v4/`

---

## 🔬 Key Enhancements over V3
1. **Ultra-Clean True Positive Recovery (+56,145 pairs):**
   - Cross-referenced V3 predictions with our ensemble model predictions (tri-ensemble of XGBoost, LightGBM, CatBoost).
   - Scanned for high-probability pairs ($p \ge 0.9995$) that were missed by V3's $\tau=0.80$ cutoff.
   - Enforced hard integer address matching: candidates were only added if their street/plot/unit numbers matched S1 exactly.
2. **Ground Truth Ceiling Constraint (Max 11 Cap):**
   - Ground truth analysis revealed that the maximum number of matches for any S1 entity in the entire dataset is strictly 11.
   - Any entity predicted with >11 matches had excess low-confidence candidates pruned to cap at 11.
