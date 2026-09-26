# Amazon ML Challenge 2026 — Business Entity Resolution

Entity resolution across three independent data sources. Given a deduplicated
Source 1, find every matching Source 2 and Source 3 record.

## Measured results

| Metric | Value | Where measured |
|---|---|---|
| Macro F₀.₅ | **96.03%** | `val_sample` (12,616 S1 entities, 518k pool), out-of-fold |
| Blocking pair recall | 95.18% | same |
| Blocking entity full coverage | 86.04% | same |
| Oracle macro F₀.₅ (ceiling) | 98.26% | same |

Macro F₀.₅ is the leaderboard metric: computed per S1 entity, then averaged.
A true singleton predicted empty scores 1.0; predicted non-empty scores 0.0.

**These are the only measured numbers in this repository.** Earlier phase
documents made projected claims of 95–98.5% that were never measured; they
are kept unedited in `docs/archive/` and should not be read as results.

## Layout

```
code/business_entity_resolution/src/   pipeline source (submission layout)
  config.py                paths, hyperparameters, feature registry
  data_loader.py           loading + entity-level disjoint train/val split
  normalize.py             text normalization, legal suffixes, transliteration
  blocking.py              5-channel candidate generation
  features.py              53 pairwise features
  model.py                 XGBoost + LightGBM + CatBoost ensemble, GroupKFold
  threshold_optimizer.py   reference (slow) 2D grid search
  threshold_optimizer_fast.py  vectorised equivalent, 29x faster
  decision_rule.py         threshold / relative / expected-F₀.₅ rules
  pairing.py               OOF-to-candidate join
  evaluate.py              macro F₀.₅ and per-entity error analysis
  consistency.py           query-exclusivity conflict resolution
  pipeline.py              end-to-end orchestration

train_model.py             train on a stratified sample, persist models + OOF cache
run_test_inference.py      memory-bounded test inference (per-country, chunked)
diagnose_recall.py         oracle ceiling and blocking-vs-scoring split
smoke_test.py              blocking recall + metric sanity check
tools/run_guarded.py       memory watchdog (lifted from reference project 9)
tests/                     equivalence tests for the fast paths
```

## Reproduce

```bash
pip install -r code/business_entity_resolution/requirements.txt

# train on a stratified sample, save models + OOF cache
python train_model.py --sample-frac 0.25

# blocking recall and metric check
python smoke_test.py

# where is F₀.₅ actually lost: blocking or scoring?
python diagnose_recall.py

# test inference, one country at a time
python tools/run_guarded.py --min-avail-mb 1800 --log run.log -- \
    python -u run_test_inference.py
```

## Design notes

**Entity-level splits.** Train/val are split on S1 entity, never on pairs, so
no entity contributes labels to two folds. `data_loader.get_entity_level_split`
asserts the two id sets are disjoint.

**Macro F₀.₅, not accuracy.** The metric is precision-weighted (β=0.5), so
losing 10 points of recall costs ~2.2 F₀.₅ while losing 10 points of
precision costs ~8.2. `evaluate.py` reproduces the worked example in the
problem statement exactly (0.714) and scores a correct singleton as 1.0.

**Sampling the training set.** 2.2M S1 entities at 35 candidates each is 77M
pairs — a 16.4 GB feature matrix. Sampling is stratified by country and
match-count bucket so the 5.58% singleton rate is preserved; an unstratified
sample would tune the threshold against the wrong distribution.

**Test inference is per-country.** All 7,638,365 matched id occurrences in
`train_ground_truth.tsv` belong to exactly one Source 1 entity and none
crosses a country boundary, so partitioning by country is lossless, and peak
index memory becomes the largest single country instead of all 9.97M records.

**The fast paths are proven, not assumed.** `threshold_optimizer_fast` and
`decision_rule` are each compared against a naive loop implementation over
~2,000 randomised cases including singletons, empty candidate sets and
probability ties; both report a maximum absolute difference of 0.0 and
identical prediction sets.

## Prediction-time guards

France appears only in test (train is US and India) and French names are often
`<city> <word>`, which makes the model over-merge businesses sharing a city
name. Two guards, both from the reference project that reports 0.9545:

- unseen-country matches must clear `threshold + 0.10`
- at most 11 matches per S1 entity, the maximum observed in training

## Submission

`output/matching_results.tsv` (scored) and `output/candidate_pairs.tsv`
(audit). Every test S1 entity must have a row; singletons get an empty field.
Validate with:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
