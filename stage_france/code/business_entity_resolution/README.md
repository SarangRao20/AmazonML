# Business Entity Resolution Pipeline

This repository contains the complete, reproducible source code for the Amazon ML Challenge 2026.

## 1. Directory Structure

```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # Text normalization, legal suffix & domain stripper
│   ├── blocking.py          # Country-partitioned multi-pass inverted index blocking
│   ├── features.py          # RapidFuzz pairwise feature extraction
│   ├── metrics.py           # Macro-averaged F_0.5 evaluator
│   ├── train_model.py       # LightGBM training & threshold optimization
│   └── inference.py         # End-to-end test inference generator
├── requirements.txt         # Pinned dependencies
└── README.md                # Reproduction guide
```

## 2. Environment Setup

Create and activate virtual environment, then install pinned dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. End-to-End Reproduction Workflow

### Step 1: Model Training & Threshold Optimization
Train the LightGBM classifier on the training split, compute feature importances, and optimize decision threshold for macro $F_{0.5}$:

```bash
PYTHONPATH=code/business_entity_resolution python3 -m src.train_model
```
*Trained model will be saved to `models/lgbm_matcher.txt`.*

### Step 2: Test Set Inference & Candidate Generation
Run country-partitioned blocking, extract pairwise features, score candidates, and generate both required output files:

```bash
PYTHONPATH=code/business_entity_resolution python3 -m src.inference \
    --test-dir dataset/test \
    --model-path models/lgbm_matcher.txt \
    --output-dir output \
    --threshold 0.65
```

This generates:
1. `output/matching_results.tsv` — Scored on public/private leaderboard
2. `output/candidate_pairs.tsv` — Candidate blocking set fed to the model

### Step 3: Validate Outputs
Verify formatting compliance using the challenge validator:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
