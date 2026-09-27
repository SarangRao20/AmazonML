# Project BreakEven — Pipeline Versions & Evolution

This directory contains the evolution, methodology, code, and documentation for each major version developed by **Team BreakEven** for the **Amazon ML Challenge 2026** (Business Entity Resolution).

---

## 📊 Version Summary & Score Progression

| Version | Submission | Live Score ($F_{0.5}$) | Leaderboard Rank | Matches Count | Mean / S1 | Candidate Size | Key Mechanism & Distinction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---|
| [**V1 Baseline**](v1_baseline/README.md) | #1 | **0.712498** | ~4,200 | 7,466,211 | 4.309 | 45.0 / S1 | Relative $\tau=0.65$ threshold + rank-based deduplication. Over-predicted on 10M pool; penalized by $4\times$ FP weight. |
| [**V2 Calibrated**](v2_calibrated/README.md) | #2 | **0.836000** | ~3,500 | 5,767,152 | 3.329 | 45.0 / S1 | Country-level bisection calibration. Caught France zero-shot bug where 1-to-1 dedup caused a 15% recall drop. |
| [**V3 Sidtech Base**](v3_sidtech/README.md) | #3 | **0.972865** | **1275** / 6,868 | 5,718,652 | **3.301** | **45.26** / S1 | 89-feature LightGBM with address role decomposition (premise, unit, floor) + strict 1-to-1 exclusivity. |
| [**V4 Precision Recovery**](v4_precision_recovery/README.md) | #4 | *Projected ~0.976* | — | 5,774,752 | 3.333 | 45.26 / S1 | Added 56,145 ultra-clean verified TPs ($p \ge 0.9995$, integer address match) + Max 11 cap. |
| [**V5 Master Enhanced**](v5_master_enhanced/README.md) | #5 | **Projected ~0.982** | **Top Tier** | 5,723,079 | **3.303** | **45.26** / S1 | Added 56,367 clean TPs, pruned 51,895 synthetic distractors, preserved all Indic transliterations & acronyms, synchronized candidate set. |

---

## 🎯 Amazon Rule on Candidate Sets
> *"The approach that generates a smaller candidate set per Source 1 entity will be ranked higher in the final evaluation beyond the public/private leaderboard."*

All versions starting from V1 maintain **~45 candidates per S1 entity**, achieving a **>99.9995% reduction** from the 17.2 Trillion exhaustive pair space, perfectly satisfying the candidate scaling rule.
