# Dense retrieval is not the lever here — measured, not assumed

Run 27 Sep 2026, Colab `dense_probe.ipynb`. Sample: 3,000 India train S1
entities, all of their true matches kept in the pool so recall stays
measurable, plus 120,000 filler pool records. 45 candidates/S1 for both
channels. TF-IDF baseline is the shipped `SparseBlocker` at the production
shard settings, not a reimplementation.

    TF-IDF alone     pair recall  97.94%
    dense alone      pair recall  96.82%
    UNION            pair recall  99.34%   (+1.40)
    dense NEW cands  precision     0.13%   (155 true of 118,257 added)

## Why this is not a near miss

The union figure is the trap. Recall does rise, 97.94 -> 99.34, and that
alone would look like a clear win. But those 118,257 added candidates
contain 155 true matches. Merged as a channel, every entity would carry
roughly 39 dense candidates of which about 0.05 are real, so:

  * the scoring set grows about 2x, and
  * F0.5 = 5R / (4P + T) charges four units for a false positive against
    one for a missed true positive.

Roughly 39 guaranteed false positives per entity is a large precision loss
bought with a small recall gain, so the expected effect on F0.5 is
negative. A true merge would need dense k cut to ~3 with a cross-encoder
reranker on top, and 0.13% precision says that reranker would have almost
nothing to promote.

## What this rules out

The standard advice for entity resolution at this scale is "add a dense
embedding channel". The reference projects here that scored well did not
do that: `reference_projects_9` (0.9545 verified on the public
leaderboard) and `ayan_multiview` are both lexical sparse retrieval plus
gradient boosting. This experiment is the evidence that the remaining
~2% recall gap on val_sample is not the kind of gap embeddings close.

## What it does not rule out

  * Dense features as *scoring* features rather than a retrieval channel
    was not tested. A pairwise cross-encoder over the existing 45
    candidates is a different experiment with a different cost profile.
  * The measurement is one country, one sample size, one model family. It
    is decisive against "merge dense into blocking" at this sample size,
    not a general claim about embeddings.
  * `intfloat/multilingual-e5-small` was chosen for transliteration. A
    domain-finetuned bi-encoder might have higher precision, but nothing
    here suggests the ceiling is reachable that way within the remaining
    budget.
