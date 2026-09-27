# Amazon ML Challenge 2026 — Business Entity Resolution
**Team BreakEven** | Sarang Rao, Pranav Sanjay Men  
**Competition Status:** Active (Deadline: 27 Sep 2026, 23:59 IST)  
**Verified Leaderboard Score:** **`0.972865`** (Rank **1275** / 6,868)  
**Master Enhanced V5 Projected Score:** **`~0.982`**

---

## 🏆 Submissions & Versions Evolution

Detailed version documentation, code, metrics, and failure/success analysis are archived in [`versions/`](versions/README.md):

| Version | Status / Score | Rank | Matches | Singletons | Details & Code |
| :--- | :---: | :---: | :---: | :---: | :---|
| **[V1 Baseline](versions/v1_baseline/README.md)** | `0.712498` | ~4,200 | 7,466,211 | 4.10% | Relative $\tau=0.65$ cutoff on 10M pool over-predicted matches; $4\times$ FP penalty bounded score. |
| **[V2 Calibrated](versions/v2_calibrated/README.md)** | `0.836000` | ~3,500 | 5,767,152 | 5.20% | Country bisection calibration. Uncovered France 1-to-1 dedup collision collapse. |
| **[V3 Role-Decomposed](versions/v3_role_decomposition/README.md)** | **`0.972865`** | **1275** | 5,718,652 | 5.84% | Direct-evidence LightGBM (89 features, premise/unit/floor roles, $\tau=0.80$, strict 1-to-1). |
| **[V4 Precision Recovery](versions/v4_precision_recovery/README.md)** | *~0.976 (Testing)* | — | 5,774,752 | 5.71% | Added 56,145 ultra-clean verified TPs ($p \ge 0.9995$, integer address match) + Max 11 cap. |
| **[V5 Master Enhanced](versions/v5_master_enhanced/README.md)** | **`~0.982 (Ready)`** | **Top Tier** | 5,723,079 | **5.76%** | **Recommended Final.** Prunes 51,895 synthetic distractors, protects Indic/acronyms, adds 56,367 clean TPs. |

Complete submission ledger: [`SUBMISSION_TRACKER.md`](SUBMISSION_TRACKER.md).

---

## 📂 Repository Structure

```
AmazonML/
├── code/business_entity_resolution/src/   # Official submission pipeline package
│   ├── config.py                          # Hyperparameters & paths
│   ├── normalize.py                       # Normalization, legal suffixes, transliteration
│   ├── blocking.py                        # Multi-channel candidate generator (~45 cands/S1)
│   ├── features.py                        # 89 fine-grained lexical & address features
│   ├── model.py                           # LightGBM / XGBoost / CatBoost ensemble models
│   ├── decision_rule.py                   # Calibrated decision rules & conflict resolution
│   └── evaluate.py                        # Macro F_0.5 evaluator
│
├── tools/                                 # Operational scripts & tools
│   ├── generate_master_v5.py              # Reproducible generator for Master Enhanced V5
│   ├── package_final_submission.sh        # Packages official zip archive (<500MB)
│   └── enforce_one_to_one.py              # Target exclusivity conflict resolver
│
├── versions/                              # Self-contained version archives & documentation
│   ├── README.md                          # Evolution summary
│   ├── v1_baseline/                       # V1 code & error diagnosis
│   ├── v2_calibrated/                     # V2 bisection calibration & France discovery
│   ├── v3_role_decomposition/             # V3 0.9728 benchmark
│   ├── v4_precision_recovery/             # V4 precision additions
│   └── v5_master_enhanced/                # V5 master enhancement implementation
│
├── tests/                                 # Unit & integration tests
├── utils/validate_submission.py           # Official competition validator
├── Documentation_template.md              # Official technical documentation
└── SUBMISSION_TRACKER.md                  # Master submission tracking log
```

---

## ⚡ Reproduce Master Enhanced V5

### 1. Generate Matching & Candidate Files
```bash
python tools/generate_master_v5.py
```
This generates:
- `output_enhanced_v5/matching_results.tsv` (1,732,544 rows, 0 collisions, 0 cross-country errors)
- `output_enhanced_v5/candidate_pairs.tsv` (45.26 candidates / S1, strictly superset of matches)

### 2. Validate Against Competition Rules
```bash
python utils/validate_submission.py \
    --matching output_enhanced_v5/matching_results.tsv \
    --candidate output_enhanced_v5/candidate_pairs.tsv \
    --test-dir dataset/test
```
*Expected Output: `PASS — no blocking issues found. Safe to submit.`*

---

## 📦 Final Submission Package
The official package for the hackathon portal:
- **Archive:** `BreakEven_submission_v5.zip` (484 MB)
- **Baseline Backup:** `BreakEven_submission_0.9728.zip` (454 MB)
- **Candidate Size Metric:** **45.26 candidates per Source 1 entity** (>99.9995% search space reduction from 17.2 Trillion pairs).
