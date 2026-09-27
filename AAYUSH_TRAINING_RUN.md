# Training run (Sep 26–27, 2026)

Local run on i3-1315U (8 threads, 7GB RAM) with `~/code/ML_DL/.venv` (Python 3.12).

## Pipeline

Country-partitioned 5-channel blocking (top-35) → 53 features (RapidFuzz +
Jaccard + numeric/blocking context) → tri-ensemble (XGB 40% / LGB 35% /
CatBoost 25%, 3-fold GroupKFold, entity-disjoint) → 2D score × margin
threshold on macro F₀.₅ + singleton gating (`apply_thresholds`: empty when
best prob < margin).

## Result

- Train: 50k stratified S1 sample (country × singleton) vs ~670k sampled pool
  (all true matches + 3 distractors/match).
- Blocking recall 94.69% (163,586/172,759 pairs).
- **Macro F₀.₅ 95.64% @ score 0.72 / margin 0.05** (`checkpoints/thresholds.json`).
- Models: `models/{xgboost,lightgbm,catboost}_model.pkl`.

## Reproduce / resume

```bash
PYTHONFAULTHANDLER=1 FEAT_WORK=4 INFER_CHUNK=12500 SAMPLE_N=50000 \
  ~/code/ML_DL/.venv/bin/python -u overnight.py
```

- Thresholds present → training skipped, resumes inference per chunk
  (`checkpoints/infer_done.json`).
- `watchdog.sh`: 2-min death checks, self-heals (halve workers/chunks), max 4 restarts.
- Inference: per-country pool index, 12.5k-S1 chunks, appends to `output/`.
- Validate: `python utils/validate_submission.py --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir dataset/test`

`output/*.tsv` are placeholders until inference completes. Dataset (2.4GB) not
in git — see Drive `AmazonML/dataset/`.
