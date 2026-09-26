"""Diagnose where macro F_0.5 is actually being lost.

Motivation
----------
The decision-rule experiment showed neither a relative per-entity floor nor
expected-F_0.5 top-k selection beats a flat threshold, and that the margin
threshold is completely inert (identical F_0.5 across margin 0.01-0.36).
That pattern indicates precision is already saturated, so F_0.5 is
recall-limited.

If F_0.5 is recall-limited, the question that decides where the remaining
time goes is: for true matches that the run failed to predict, was the
match

  (a) never retrieved as a candidate          -> blocking problem, or
  (b) retrieved but scored below the threshold  -> scoring problem?

Those point at completely different work. This script measures the split and
reports the oracle ceiling: the F_0.5 a *perfect* classifier would achieve
on the candidate set actually produced. No decision rule can beat it.

Inputs
------
Reads the cache written by ``pipeline._cache_oof``, which stores the OOF
probabilities together with the (s1_id, s2_s3_id) pair keys. The keys are
essential: OOF rows and blocking candidate lists are in different orders,
so joining them positionally would silently pair candidates with the wrong
probabilities.

Usage
-----
    python -m code.business_entity_resolution.src.pipeline --mode train-only
    python diagnose_recall.py
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from code.business_entity_resolution.src.config import OOF_CACHE
from code.business_entity_resolution.src.decision_rule import EntityIndex


def build_probabilities(payload) -> dict:
    """Return the S1 -> [(candidate, prob)] grouping from the cache.

    Prefers the grouping the training run already computed, because that is
    the one produced by ``pairing.build_probabilities_by_s1`` after realigning
    on ``row_index``. Re-deriving the join here is how the first version of
    this script ended up reporting 18% instead of 96%: the OOF frame is in
    fold order, so a positional zip against the feature arrays pairs every
    candidate with another entity's probability.
    """
    if "probabilities_by_s1" in payload:
        return payload["probabilities_by_s1"]

    oof = payload["oof"]
    if "row_index" not in oof.columns:
        raise SystemExit(
            "Cache has neither 'probabilities_by_s1' nor an 'row_index' "
            "column, so predictions cannot be joined to their candidates. "
            "Re-run train_model.py.")
    from code.business_entity_resolution.src.pairing import (
        build_probabilities_by_s1,
    )
    return build_probabilities_by_s1(oof, payload["s1_id"], payload["s2_s3_id"])


def oracle_macro_f_beta(candidate_ids_by_s1, gt_dict) -> float:
    """F_0.5 of a perfect classifier on the candidate set we produced.

    The perfect decision for each entity is to predict exactly its true
    matches that blocking actually retrieved, and nothing else. Two cases
    need care:

    * a true singleton is scored 1.0 because the correct prediction is the
      empty set, regardless of how many candidates were retrieved — a
      classifier that knew the answer would simply predict nothing;
    * a non-singleton with no retrieved true match is stuck at 0.0 no
      matter what, because blocking never surfaced the match.

    The result is an upper bound on every possible decision rule.
    """
    total = 0.0
    for s1_id, truth in gt_dict.items():
        if not truth:
            # Correct decision is "no matches", which scores 1.0.
            total += 1.0
            continue
        got = candidate_ids_by_s1.get(s1_id, set())
        tp = truth & got
        if not tp:
            total += 0.0
            continue
        precision = 1.0
        recall = len(tp) / len(truth)
        total += (1.25 * precision * recall) / (0.25 * precision + recall)
    return total / max(len(gt_dict), 1)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof-cache", type=Path, default=OOF_CACHE,
                    help="Path to the OOF cache written by the pipeline")
    args = ap.parse_args()
    cache_path = args.oof_cache

    print("=" * 74)
    print("RECALL CEILING DIAGNOSIS")
    print("=" * 74)

    if not cache_path.exists():
        print(f"\nNo OOF cache at {cache_path}.")
        print("Run the pipeline first:\n"
              "  python -m code.business_entity_resolution.src.pipeline "
              "--mode train-only")
        return 1

    with open(cache_path, "rb") as fh:
        payload = pickle.load(fh)
    oof = payload["oof"]
    gt_dict = payload["gt_dict"]
    print(f"\n[1] Cache loaded")
    print(f"    OOF rows            : {len(oof):,}")
    print(f"    ground truth entities: {len(gt_dict):,}")

    probabilities = build_probabilities(payload)
    n_pairs = sum(len(v) for v in probabilities.values())
    n_entities = len(probabilities)
    print(f"    scored pairs         : {n_pairs:,}")
    print(f"    S1 entities with candidates: {n_entities:,}")

    candidate_ids = {s1: {c for c, _ in cands} for s1, cands in probabilities.items()}

    # ---- 1. Oracle ceiling -------------------------------------------
    print("\n[2] Oracle ceiling (perfect classifier on OUR candidate set)")
    total_true = 0
    captured = 0
    full_cover = 0
    non_single = 0
    for s1_id, truth in gt_dict.items():
        if not truth:
            continue
        non_single += 1
        got = candidate_ids.get(s1_id, set())
        total_true += len(truth)
        captured += len(truth & got)
        if truth.issubset(got):
            full_cover += 1

    pair_recall = captured / total_true if total_true else 0.0
    full_cov = full_cover / non_single if non_single else 0.0
    oracle = oracle_macro_f_beta(candidate_ids, gt_dict)

    print(f"    true pairs                    : {total_true:,}")
    print(f"    retrieved by blocking         : {captured:,}")
    print(f"    PAIR recall                   : {pair_recall * 100:.2f}%")
    print(f"    ENTITY full coverage          : {full_cov * 100:.2f}%"
          f"  ({full_cover:,}/{non_single:,})")
    print(f"    ORACLE macro F_0.5            : {oracle * 100:.2f}%")
    print(f"      ^ hard ceiling; no decision rule can exceed this")

    # ---- 2. Where the missing true matches are ------------------------
    print("\n[3] Splitting the loss: blocking vs scoring")
    index = EntityIndex(probabilities, gt_dict)

    lost_to_blocking = 0     # true match never appeared as a candidate
    lost_to_scoring = 0      # retrieved, but no threshold kept it
    recall_at_best = 0.0
    precision_sum = 0.0
    scored_entities = 0

    # Evaluate the actual deployed rule so the two loss buckets are measured
    # against what the pipeline really does.
    from code.business_entity_resolution.src.decision_rule import DecisionRuleOptimizer
    opt = DecisionRuleOptimizer(beta=0.5)
    kept = opt.kept_for_threshold(index, 0.70, 0.01)
    tp = index._true_positives_for(kept)
    f = index._f_beta(kept, tp)
    measured = float(f.mean())

    for i, s1_id in enumerate(index.entity_ids):
        truth = index.n_true[i]
        if truth == 0:
            continue
        s, e = index.start[i], index.start[i + 1]
        retrieved_tp = index.cum_tp[e - 1] if e > s else 0.0
        missed = truth - retrieved_tp
        if missed > 0:
            lost_to_blocking += int(missed)
        # Retrieved but dropped by the threshold
        lost_to_scoring += int(round(index.n_true[i] - tp[i])) if tp[i] < truth else 0
        recall_at_best += float(tp[i]) / truth
        precision_sum += float(tp[i]) / kept[i] if kept[i] > 0 else 0.0
        scored_entities += 1

    n = max(scored_entities, 1)
    print(f"    true matches never retrieved  : {lost_to_blocking:,}"
          f"   ({100 * lost_to_blocking / max(total_true, 1):.2f}% of all true pairs)")
    print(f"    true matches dropped by rule  : {lost_to_scoring:,}")
    print(f"    macro recall at score=0.70    : {100 * recall_at_best / n:.3f}%")
    print(f"    macro precision at score=0.70 : {100 * precision_sum / n:.3f}%")
    print(f"    measured macro F_0.5          : {measured * 100:.4f}%")
    print(f"    oracle macro F_0.5            : {oracle * 100:.4f}%")
    print(f"    headroom left by the rule     : "
          f"{(oracle - measured) * 100:.4f} points")

    print("\n" + "=" * 74)
    blocking_share = lost_to_blocking / max(total_true, 1)
    print(f"VERDICT  ceiling {oracle * 100:.2f}%  |  "
          f"blocking loses {100 * blocking_share:.2f}% of true pairs")
    if blocking_share > 0.10:
        print("  -> BLOCKING is the dominant loss. Raise the per-entity")
        print("     candidate cap and add retrieval channels (TF-IDF name/addr/char)")
        print("     before touching features or the model.")
    else:
        print("  -> Blocking is adequate. The remaining loss is in scoring and")
        print("     decisioning, so invest in features or the model instead.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
