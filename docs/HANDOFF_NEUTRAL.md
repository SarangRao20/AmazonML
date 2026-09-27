# Context: Amazon ML Challenge 2026 — Entity Resolution

**Purpose of this document.** To hand a complete factual record to a second
reasoner. It is deliberately written as measurements and observations only.
There are no recommendations, no diagnosis, and no framing that would steer
a conclusion. Where something is unknown it says so. Any inference drawn from
these numbers is the reader's to make.

Timestamp of record: 27 Sep 2026, 15:05 IST. Deadline 23:59 IST (8h 53m
remaining at time of writing).

---

## 1. The task

Given three record sources, for every Source-1 entity, predict which
Source-2 / Source-3 records refer to the same real-world business. A Source-1
entity may match zero, one, or many.

Each source file is tab-separated with columns `entity_id`, `business_name`,
`business_address`, `country`. Source is indicated by the id prefix
(`S1-`, `S2-`, `S3-`).

Training covers US and India. Test additionally contains **France, which does
not appear in training at all.** The problem statement explicitly instructs
treating country as an open set of labels and not hard-coding the pipeline to
`{US, India}`.

### Metric

```
F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
      = 5*TP / (T + 4*P)
```

Computed as a macro average: F_0.5 is calculated per Source-1 entity, then
averaged across all Source-1 entities. Singletons are included in that
average: an entity with no true matches scores **1.0** if predicted empty and
**0.0** if any match is predicted.

False positives are weighted 4× relative to misses. The formula's arithmetic
consequence, which recurs below: if the true count per entity is `T` and we
predict `P` pairs with perfect precision (every predicted pair correct),
then

```
F_0.5 = 5*T / (T + 4*P)      (capped at 1.0)
```

so the score is bounded above by the predicted count alone, independent of
model quality.

### Required output

- `matching_results.tsv` — scored on the leaderboard.
- `candidate_pairs.tsv` — the blocking candidate set, **the last stage before
  the model**, i.e. exactly what inference runs over. Every id in
  `matching_results.tsv` should appear here. Not scored, but reviewed by
  organisers, who state that a *smaller* candidate set per S1 ranks higher in
  final evaluation beyond the leaderboard score.

A validator ships with the challenge: `utils/validate_submission.py`.

---

## 2. Data facts (measured)

| Quantity | Value |
|---|---|
| Train S1 | 2,206,821 |
| Train S2 | 5,034,616 |
| Train S3 | 5,285,603 |
| Train ground truth pairs | 7,638,365 |
| **Distinct targets in ground truth** | **7,638,365** |
| **Targets claimed by >1 S1** | **0** |
| Test S1 | 1,732,544 |
| Test S2 + S3 (pool) | ~10M |
| Test S1 by country | US 663,106 · India 809,986 · France 259,452 |
| Singletons in train | 123,247 (5.58%) |

Ground truth is a **perfect matching** — no Source-2/3 record is ever matched
to two Source-1 records. This was verified directly by counting distinct
targets against pair count over the full 7,638,365 rows.

### Per-country train prior (measured)

| Country | S1 | true mean | empty % | p50 | p90 | max |
|---|---|---|---|---|---|---|
| India | 883,188 | 3.46 | 5.59 | 3 | 6 | 11 |
| US | 1,323,633 | 3.46 | 5.58 | 3 | 6 | 11 |

The two countries have an **identical** distribution. France has no train
prior at all, since it is absent from training.

Not one of the 7,638,365 ground-truth pairs crosses a country boundary
(verified), so country partitioning is lossless.

---

## 3. Current pipeline

```
blocking   sharded sparse multi-channel TF-IDF, 4 channels + exact key
           ~45 candidates per S1, capped
features   53 features: RapidFuzz string similarity, token overlap, Jaccard,
           house numbers, postal hierarchy, landmarks, cross-channel rank and
           agreement, density features, reciprocity
model      XGBoost + LightGBM + CatBoost ensemble, weights
           {xgb 0.4, lgbm 0.35, catboost 0.25}
decision   per-country threshold, then per-S1 cap of 11
```

Text normalisation expands English legal suffixes (`corp`→`corporation`,
`pvt`→`private`, `ltd`→`limited`, `inc`→`incorporated`, `llc`→…).
**French legal suffixes are not handled.** Observed French S1 names include
`ZNB Club SARL`, `Thermal & Fils SASU`, `Elephant Centre EURL`,
`Saint-Herblain Societe SARL`, `Association de Pena`.

### Training data actually used by the current model

```
12,616 S1 entities (out of 2,206,821 available)   =  0.57%
96,696 candidate rows
pool used: ~5.1M records
```

---

## 4. Measurements on labelled data (out-of-fold)

The model produces out-of-fold predictions covering 12,616 S1 entities **with
true labels**. Threshold sweep, macro F_0.5, whole-S1 count in denominator:

| tau | F_0.5 | mean | empty % |
|---|---|---|---|
| 0.50 | 0.9631 | 3.38 | 6.26 |
| **0.70** | **0.9668** | **3.26** | **6.71** |
| 0.80 | 0.9663 | 3.20 | 6.91 |
| 0.90 | 0.9602 | 3.10 | 7.28 |
| 0.95 | 0.9500 | 3.00 | 7.70 |
| 0.97 | 0.9377 | 2.89 | 8.31 |
| 0.99 | 0.9094 | 2.69 | 9.79 |
| 0.995 | 0.8968 | 2.61 | 10.42 |
| 0.999 | 0.7677 | 1.97 | 19.42 |

Truth on this set: mean 3.447, empty 6.46%. The maximum over the sweep is
**0.9668 at tau≈0.70**. Every tau stricter than ~0.7 scores lower. A finer
grid puts the peak at tau≈0.70, F_0.5 0.9667.

Other OOF results:
- 1-to-1 target deduplication: **+0.0005** (0.9667 → 0.9672).
- Per-entity "expected F0.5" decision rule (below): **0.9654**, i.e. *worse*
  than the plain threshold's 0.9668. Swept over 49 values of its one
  parameter, best at shift −0.25.

### The per-entity expected-F0.5 rule, for reference

For one entity with recalibrated candidate probabilities `q_1..q_n`:

```
keep none : E[F] = prod_i (1 - q_i)                     (probability it is a singleton)
keep k    : E[F] = 1.25 * (sum of top-k q) / (0.25 * sum of all q + k)
```

Pick the k maximising E[F]; empty the entity if "none" beats it.
`q = sigmoid(logit(p) + shift)`, `shift` tuned out of fold.

---

## 5. Test-set predictions and leaderboard results

### Submission 1 — raw run

Predicted distribution, per country:

| Country | S1 | pairs | mean | empty % |
|---|---|---|---|---|
| France | 259,452 | 1,703,855 | 6.57 | 0.16 |
| India | 809,986 | 4,035,350 | 4.98 | 1.11 |
| US | 663,106 | 3,267,978 | 4.93 | 1.20 |

France had 15.57% of entities at the cap of 11. Overall mean 5.626, empty
1.15%.

**Leaderboard: 0.712498.**

### Submission 2 — per-country calibration

Thresholds solved by bisection so predicted mean lands on 3.46:

```
France  tau=0.99027  mean 3.460  empty 5.12%
India   tau=0.96887  mean 3.460  empty 5.31%
US      tau=0.97444  mean 3.460  empty 5.25%
```

5,994,576 pairs, mean 3.460, validator PASS.

The three solved thresholds differ by up to 0.02. France's scores sit lower
for equivalent evidence — consistent with it being unseen in training.

**Leaderboard: 0.836.**

### Post-processing applied to submission 2 (not submitted)

1-to-1 target uniqueness, ties broken on probability:

```
pairs    5,994,576 -> 5,767,152  (227,424 removed)
mean     3.460 -> 3.329
empty    5.25% -> 5.75%  (99,670 rows, 3,350 newly emptied)
max/S1   11
1-to-1 verified: 5,767,152 distinct targets, 0 duplicated
validator: PASS
```

---

## 6. Arithmetic on the observed scores

For a country whose true mean is 3.46, the ceiling given predicted count `P`
and perfect precision is `5*3.46/(3.46+4*P)`:

| P | ceiling |
|---|---|
| 6.57 (France, submission 1) | 0.582 |
| 4.98 (India, submission 1) | 0.740 |
| 4.93 (US, submission 1) | 0.746 |
| 3.33 (submission 2) | 1.000 (uncapped) |

Submission 1 scored 0.712 against a computed macro ceiling of 0.689 — i.e.
at or slightly above it, with France's true mean unknown and possibly higher
than 3.46.

Inverting the formula for submission 2 at P=3.329, T=3.46, F_0.5=0.836 gives
TP ≈ 2.81, i.e. **recall ≈ 81%, precision ≈ 84%.** This inversion assumes
T=3.46 for every country including France, which is unverified.

---

## 7. Reference projects available in the repo

22 projects under `references/` and `references2/`. Documented results found:

**`references/reference_projects_9`** — multi-channel TF-IDF blocking +
stage-1 pruner + LightGBM, with a per-entity expected-F0.5 rule.
README states: *"On a 16 GB laptop (the configuration that produced
**leaderboard 0.9545**)"*. Its run command uses `--train-pct 30`, i.e. 30% of
the full train pool rather than a small validation sample. No blocking-recall
figure is documented in its README. It selects between the global threshold
and the entity rule by whichever scores higher out of fold.

**`references/reference_projects_12`** — keeps a running experiment log.

```
EXP-002  val F_0.5 0.8722   blocking recall ceiling 86.21%
EXP-003  val F_0.5 0.8710   PUBLIC LEADERBOARD 0.826
```

Its own note: *"Blocking recall ceiling at 86.21% is below the 95% gate."*
Its blocking was token + 4-prefix + Soundex + postal index — weaker than the
current pipeline's multi-channel TF-IDF with character n-grams. It reports
precision 1.0000 and recall 0.8600 at threshold 0.93, and attributes its
limitation to blocking recall.

**Earlier in this session, a dense-embedding channel was evaluated and
rejected** (recorded in `docs/DENSE_CHANNEL_NEGATIVE.md`): TF-IDF candidate
recall 97.94%, dense 96.82%, union 99.34%, but pairs added *only* by the
dense channel had precision 0.13% — 155 true matches among 118,257 added
pairs. No full dense build was attempted.

### Current leaderboard (as observed)

Rank 1: 0.991811. Rank 2: 0.991483. Rank 3: 0.990881. Ranks 4–28: 0.990–0.991.
6,868 teams total. Our 0.836 placed around rank 3,500.

---

## 8. Blocking recall — status: NOT YET MEASURED

Reference project 12 states its leaderboard score tracked its blocking recall
ceiling. For this pipeline that number **has not been measured**.

A first attempt returned 99.95% pair recall but was **invalid**: the pool had
been filtered down to only the true matches of the sampled S1 entities
(6,410 records instead of 6.19M), which makes blocking trivial.

A corrected run against the full per-country pool failed on an import error
(`code` is not an importable package without `PYTHONPATH` set) and did not
execute. No valid figure exists yet.

This is the outstanding measurement most directly relevant to whether the
model or the blocker is the binding constraint.

---

## 9. Retrain on the full pool — status: NOT COMPLETED

A retrain was launched on a Colab T4 session with the intent of training on
the full per-country pool (10,320,219 records) rather than the 5.1M sample.
Configuration: `--sample-frac 0.08`, i.e. 176,546 S1 entities.

- First attempt failed: `AttributeError: 'DataFrame' object has no attribute
  'height'` — the Colab copy of the code was stale relative to the local
  repository, which already used `len(s2)`. Patched by uploading current code.
- Second attempt reached the blocking phase: India pool 4,133,346 records,
  `name_word` channel in 346s, then the character channel.
- The Colab session has since terminated. No models were produced. No OOF
  number from the retrained model exists.

Measured blocking timings from that run, recorded for whoever repeats it:

```
Colab, India, 70,426 S1, 4.13M pool:  name_word 346s (5 shards)
Local,  India, 809,986 S1, 4.72M pool: name_word 403s
```

11× fewer S1 entities at nearly the same wall time, which indicates the
blocking cost is dominated by vectorising the pool rather than by the number
of S1 queries. Untested extrapolation to larger samples.

---

## 10. Current state of outputs

| Path | Rows | Pairs | Mean | Empty % | Validator |
|---|---|---|---|---|---|
| `output_baseline/` | 1,732,544 | 0 | 0.000 | 100% | PASS |
| `output/` (submission 1) | 1,732,544 | 9,747,109 | 5.626 | 1.15% | PASS |
| `output_calibrated/` (submission 2) | 1,732,544 | 5,994,576 | 3.460 | 5.25% | PASS |
| `output_final/` (calibrated + 1-to-1) | 1,732,544 | 5,767,152 | 3.329 | 5.75% | PASS |

Probability side-files exist and cover the pairs in each submission:

```
output/parts/France.scores.tsv   5,189,040 rows (20 candidates/S1)
output/parts/India.scores.tsv    ~4.0M rows (submitted pairs only)
output/parts/US.scores.tsv       ~3.3M rows (submitted pairs only)
```

Note: France's side-file came from the Colab run and covers that run's own
top-20 blocking candidates, whereas the India/US side-files cover only the
pairs already in the submitted file. A consequence is that the France scores
contain ids that were never submitted, and calibrating against them without
intersecting with the submitted set produced 23,259 matches absent from
`candidate_pairs.tsv`. The validator caught this. Any re-use of the France
scores should intersect with the submitted set first.

These scores make re-thresholding cheap: re-deriving probabilities for the
7.3M already-submitted pairs took 25 minutes, versus 9 hours for a full
inference pass. A re-thresholded submission can therefore be produced in
roughly 40 minutes end to end without retraining.

`BreakEven_submission.zip` (450 MB) is built and contains `output/`,
`code/business_entity_resolution/{src,scripts,tools}`, `README.md`,
`requirements.txt`, and a filled `Documentation_template.md`. Its
`matching_results.tsv` is byte-identical to `output_final/` (md5
`85b03bb9fa4c79d0d01e5812a2bd10df`).

---

## 11. Constraints and operating notes

- **Submissions are limited.** Two used (0.712, 0.836). Every further
  submission should be justified by a measurement, not by a deadline.
- Full test inference took **9 hours**; a re-threshold via rescore takes
  **25 minutes**. Training is the bottleneck, not inference.
- 8h 53m remained at time of writing.
- A retrain at 8% sampling was estimated at 4.5–6 hours; that estimate was
  not confirmed, because the run terminated early.
- Machine: 15 GB RAM, 4 GB zram, 16 logical cores.
- Colab access: `colab new -s <name> --gpu T4` works; free sessions terminate
  on their own; the compute-unit balance reads 0.00.

## 12. Open questions, stated as questions

1. What accounts for the 0.13 gap between OOF 0.9668 and test 0.836 at the
   same predicted mean?
2. Does the 5.1M→10M pool difference account for it, and does retraining on
   the full pool close it, and by how much?
3. Is mean 3.46 the right target, or is a lower count better given the 4×
   penalty on false positives?
4. Is the ~81% recall implied by 0.836 consistent with an OOF recall near
   100%, and what would raise it?
5. Is 0.9918 reachable by tuning this model family, or does it require a
   different approach?
6. What is this pipeline's blocking recall ceiling against the full pool?
