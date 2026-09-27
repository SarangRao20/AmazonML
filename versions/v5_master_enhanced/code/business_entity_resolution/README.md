# Business Entity Resolution Pipeline

Source code for the Amazon ML Challenge 2026. This directory is
self-contained: together with the training and test data it regenerates both
files in `output/`.

## 1. Layout

```
code/business_entity_resolution/
├── src/                     library modules
│   ├── config.py             paths, constants, F_BETA
│   ├── data_loader.py        TSV readers (explicit sep="\t")
│   ├── normalize.py          text normalisation, legal-suffix stripping
│   ├── blocking.py           multi-pass inverted-index blocking
│   ├── blocking_sparse.py    sharded sparse TF-IDF blocker (used for the run)
│   ├── features.py           RapidFuzz + rank features, RecView, chunking
│   ├── model.py              LightGBM/XGBoost/CatBoost ensemble
│   ├── decision_rule.py      relative per-entity threshold rule
│   ├── threshold_optimizer.py, threshold_optimizer_fast.py
│   ├── pairing.py            S1/S2/S3 record pairing
│   ├── consistency.py        post-hoc consistency resolution
│   ├── evaluate.py           macro F_0.5 evaluator
│   └── pipeline.py           orchestration
├── scripts/                 entry points, run in this order
│   ├── train_model.py        trains the ensemble, writes models/ + decision_rule.json
│   ├── run_test_inference.py blocking → features → scores → output/
│   └── rescore_submitted.py  re-scores only already-submitted pairs
├── tools/                   post-processing
│   ├── calibrate_and_assemble.py   per-country threshold calibration
│   ├── enforce_one_to_one.py       1-to-1 target uniqueness
│   ├── apply_deduplication.py      rank-based 1-to-1 (superseded)
│   └── inspect_country.py          per-country distribution audit
├── requirements.txt         pinned dependencies
└── README.md                this file
```

## 2. Environment

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

All TSVs are tab-separated. Read them with `sep="\t"`; the default comma
separator silently yields one column.

## 3. Reproducing the submission

Run from the repository root. `PYTHONPATH` makes the `src` package importable.

### Step 1 — train

```bash
PYTHONPATH=code/business_entity_resolution python3 code/business_entity_resolution/scripts/train_model.py
```

Writes the three model files and `models/decision_rule.json`. Training is
per-country and streamed; the ensemble is calibrated on held-out folds and
the decision rule is tuned for macro F₀.₅.

### Step 2 — blocking, scoring, output

```bash
PYTHONPATH=code/business_entity_resolution python3 code/business_entity_resolution/scripts/run_test_inference.py \
    --test-dir dataset/test --model-dir models --out output
```

This produces `output/parts/<country>.matching.tsv` and
`<country>.candidates.tsv` per country, then concatenates them into
`output/matching_results.tsv` and `output/candidate_pairs.tsv`. Per-country
`.done` markers make it resumable; delete a marker to redo that country.

Full inference took ~9 hours on 16 cores for 1,732,544 S1 entities against
~10M S2/S3 records. Peak RSS was ~6 GB.

### Step 3 — calibrate (required, not optional)

The rule in Step 2 is tuned on held-out folds whose pool is ~5.1M records.
The test pool is ~10M, so the same threshold admits more lookalike
candidates: Step 2 alone predicts mean 4.93–6.57 matches per entity against
a true prior of 3.46, which under F₀.₅ = 5R/(T+4P) caps the score near 0.69
even with perfect precision. Re-derive probabilities for the submitted pairs
and re-solve one threshold per country:

```bash
PYTHONPATH=code/business_entity_resolution python3 code/business_entity_resolution/scripts/rescore_submitted.py
PYTHONPATH=code/business_entity_resolution python3 code/business_entity_resolution/tools/calibrate_and_assemble.py \
    --out output_calibrated
```

Thresholds must be per country. The US calibrates near τ=0.974 and France
near τ=0.990; at either value the other country is 15–30% off, because
France never appears in training and its probabilities sit lower for
equivalent evidence. A shared threshold leaves France over-committed, and a
macro average lets one country cap the whole submission.

### Step 4 — enforce 1-to-1

```bash
PYTHONPATH=code/business_entity_resolution python3 code/business_entity_resolution/tools/enforce_one_to_one.py \
    --matching output_calibrated/matching_results.tsv --out output_final
```

In the training ground truth 7,638,365 pairs claim 7,638,365 distinct
targets, so a target is never shared and every duplicate claim is a
guaranteed false positive. Ties are broken on model probability, not rank,
since rank is not comparable across entities.

### Step 5 — validate

```bash
python3 utils/validate_submission.py \
    --matching output_final/matching_results.tsv \
    --candidate output_final/candidate_pairs.tsv \
    --test-dir dataset/test
```

Must print `PASS`.

## 4. What the submitted files contain

```
output/matching_results.tsv   1,732,544 rows, mean 3.46 before 1-to-1, 3.33 after
output/candidate_pairs.tsv    ~45 candidates per S1, cap set by the blocker
```

Calibration only ever removes pairs, so every final match is still present in
`candidate_pairs.tsv`.

## 5. Notes

- Country labels are treated as open-set strings, never hard-coded to a
  fixed list; France appears only in test.
- Blocking is sharded and streamed to keep peak memory near 6 GB.
- No external data or geocoding is used anywhere in the pipeline.
