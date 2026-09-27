# Handoff — Amazon ML Challenge 2026, Business Entity Resolution

Written so a second model can reach a decision without re-deriving
anything. Where a number is estimated rather than measured, it says so.
The open questions are at the bottom and they are the point of this
document.

## Hard constraints

- **Deadline 2026-09-27 23:59 IST** (from `guidelines.pdf`: "Challenge
  Window: 25th September 2026, 12:00 AM IST to 27th September 2026,
  11:59"). The submit button is disabled after.
- Local box: 15 GB RAM, ~3 GB genuinely free (Chrome ~2.2 GB, Devin
  ~1.3 GB, opencode ~0.8 GB), 16 logical cores, 4 GB zram. Colab offers
  ~12 GB but needs a GPU runtime and the user's browser.
- `tools/run_guarded.py` is a memory watchdog that kills the job when
  available RAM drops under a floor (currently 1600 MB). **It has killed
  three runs today.** It is a feature: each kill was a real OOM avoided.
- A second runtime exists via Colab (notebooks `dense_probe.ipynb`,
  `retrain_colab.ipynb`). Colab runs server-side, so closing the browser
  does not stop it. Colab auth state was repaired during the session
  (`client_secrets.json` reconstructed from an existing `token.json`,
  access token refreshed via the refresh token).

## What is currently running, and why it must not be disturbed

`run_test_inference.py` pid 272767, launched 03:25, detached
(`setsid`, own session SID 272765 so it survives terminal and opencode
exit). Log `logs/testrun3_032459.log`. Watchdog `run_guarded.py` pid
272765, floor 1600 MB.

State: France done. India (809,986 S1 against a 4,717,565 pool) inside
`name_char`, the slowest channel, 16 shards of 400k. At 06:00 it was 2h37m
in with no new log line, because the blocker logs only on channel
completion, not within a channel.

**Do not kill it.** 6+ hours of work, and nothing else depends on it
finishing. If it dies, France is still on disk in
`output/parts/France.matching.tsv` and the `.done` markers make any rerun
resume rather than recompute.

## Safety net already in place

`output_baseline/matching_results.tsv` — 1,732,544 rows, all empty,
passes `utils/validate_submission.py`. Scores 0, but it is a **valid**
submission that can be made at any moment. So there is no scenario where
nothing can be submitted.

## The pipeline

```
train_model.py     train, save models/ + models/decision_rule.json + OOF cache
run_test_inference.py  per-country blocking -> scoring -> output/parts/<country>.*
                   assembles output/ on completion
retune.py          re-apply the decision rule to saved scores, offline
inspect_country.py audit a finished country part
check.sh           status (reads the newest logs/testrun*.log)
run_pipeline.sh    driver: retrain -> infer -> validate
```

Submission = `output/matching_results.tsv` (scored) and
`output/candidate_pairs.tsv`. Every one of the 1,732,544 S1 entities must
appear on exactly one row; a duplicate row is a hard validation error.

## Measured results, in order

| # | Metric | Value | Provenance |
|---|---|---|---|
| 1 | Macro F0.5 | 96.03% | val_sample, 12,616 S1, dict blocker |
| 2 | Macro F0.5 | **96.70%** | same 12,616, sparse blocker |
| 3 | Blocking pair recall | 95.18% → **98.11%** | same |
| 4 | Entity full coverage | 86.04% → **94.10%** | same |
| 5 | Blocking recall, all 49,999 val_sample S1 | **96.86%** | cap 45, sparse |
| 6 | Train ground truth per S1 entity | mean **3.46**, p50 3, p90 6, p99 8, **max 11**, 5.58% singletons — identical for US and India | `dataset/train/train_ground_truth.tsv` |
| 7 | France prediction, first full run | mean **6.57**, p90 11, **15.57% at the cap**, 0.16% empty | `inspect_country.py` |
| 8 | Dense probe | TF-IDF 97.94%, dense 96.82%, union 99.34%, **new-candidate precision 0.13%** (155 true of 118,257) | Colab, 3,000 India train S1 |

`MAX_PER_S1 = 11` is validated, not inherited: #6 shows the ground truth
tops out at exactly 11, so the cap cannot discard a true match.

**Macro F0.5 is the metric** (`README.md:15`), per S1 entity then
averaged; a true singleton predicted empty scores 1.0. It is
precision-weighted, `F0.5 = 5R / (4P + T)`, so a false positive costs four
times what a missed true positive costs.

## Experiment that failed, with the reasoning

Dense retrieval as a fifth blocking channel. Union recall rises 97.94 →
99.34, which looks like a win, but those 118,257 added candidates hold 155
true matches. Merged, each entity would carry ~39 dense candidates of
which ~0.05 are real, so the expected F0.5 effect is negative. Written up
in `docs/DENSE_CHANNEL_NEGATIVE.md`.

This also **corrects an earlier unevidenced claim of mine** that top teams
were probably adding embedding channels. The references that scored well
here are lexical: `references/reference_projects_9` (0.9545 verified on
the public leaderboard) and `references2/ayan_multiview`.

## What is unproven

Flagging these because most of my confident statements today were wrong.

1. **The retrain was never measured.** Training on the full per-country
   pool instead of `val_sample`'s 518k is motivated by 18 of 53 features
   being channel/rank/density counts. That is a plausible argument and no
   evidence. It has never completed a run: three OOM kills, then a fourth
   failure on a `s2.height` bug (Polars attribute on a pandas frame) that
   I introduced and never executed locally before shipping it to Colab.
2. **France's F0.5 ≈ 0.60 is inferred, not measured.** It comes from #7
   versus #6, plus `F0.5 = 5R/(4P+T)`. If France's true match count is
   genuinely near 3.46 and recall is decent, over-predicting 6.57 costs
   roughly 0.87 → 0.50. But no France labels exist and I have never scored
   France.
3. **The whole val → test extrapolation is unmeasured.** 96.70% is
   US+India only. `val_sample` has no France. My estimate that test lands
   near 0.91 assumes France ≈ 0.60 and inherits US/India unchanged.
4. **The 13:00 completion time is a projection** from an RSS growth rate
   of 5 MB/min against 62 MB of retained char hits per shard.

## Situation the user saw

The leaderboard they looked at is **"top gainers"**, not the main ranking.
It shows 99.0748, 99.029, 98.934, 98.9 for the first four. Whether that
is the main leaderboard, the public one, or the largest recent jumps is
unresolved, so it is **not** a sound basis for judging rank.

For reference, the old oracle ceiling with the dict blocker was 98.26% at
95.18% recall; at 98.11% recall the ceiling is roughly 99.3%, so a 99.07
score is inside our ceiling and is not evidence that the leaders found
something categorically different.

## The three questions

**Q1. Is the France over-prediction real, and what is the highest-value
action on it?** Evidence for: #6 versus #7, and France is unseen in train
while US and India are seen. Evidence against or qualifying: it is
inferred, never scored. If real, a re-threshold costs ~40 min (France
blocking 163 s, scoring ~37 min) because `run_test_inference.py` now
persists per-candidate scores and `retune.py` sweeps the floor offline in
seconds. If the *model* is at fault rather than the threshold, this fixes
nothing and the 40 min is wasted. Is there a way to tell those apart
before spending it? Note the current run does **not** write scores, so
France cannot be re-thresholded from its output.

**Q2. What is the single highest-expected-value use of the remaining
~17 hours?** Candidates and their costs: let the run finish and submit
(~96% est.); re-threshold France (~40 min, est. +2-3 if Q1 holds); the
full-pool retrain plus a full re-inference (~8 h, unproven); raise
`max_candidates` from 45 toward 112, which `references2/ayan_multiview`
reports as 0.989 at 45 versus 0.9905 at 112, so a very thin gain for
roughly double the scoring cost. Raising the cap is also the one lever
that raises the recall ceiling itself, and recall is what caps F0.5.
Is there a better candidate I have not considered?

**Q3. What would actually close a 2-point gap to a 99.07 leaderboard
entry?** The dense probe ruled out the standard answer. Given that
`F0.5 ≈ recall` when precision is good, the gap is substantially a
blocking-recall gap. Is there a retrieval formulation with materially
higher precision at 45 candidates than four TF-IDF channels plus an
order-invariant name key, that does not need a GPU or a trained model?
Constraints that rule things out: no FAISS, no GPU locally, no network
on the inference machine, `max_features` on the vectorisers already bounds
the vocabulary because the unmaterialised full-vocabulary dict was what
caused the OOMs.

## Notes for whoever picks this up

- Do not re-clone the reference projects. They are already on disk under
  `references/` and `references2/`, and are deliberately untracked.
- `models/oof_predictions.pkl` holds the OOF cache, 12,616 entities with
  probabilities and the tuned decision rule. Cheap to explore rule
  variants offline; it is in-sample for the incumbent in the sense that
  those entities were the training sample, so scores on it are optimistic
  for the incumbent in any head-to-head.
- The score side-file and `retune.py` landed **after** the current run
  started, which is why the run has no scores. A future run gets them for
  free.
- Seven commits on `main`, unpushed.
