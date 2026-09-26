"""Train the matching ensemble on a stratified sample and persist it.

Sampling is stratified by country x match-count bucket so the singleton rate
and the class balance of the sample match the full training set. That
matters more than it looks: singletons are 5.58% of S1 entities and each one
scores a full 1.0 or 0.0 under macro F_0.5, so a sample that distorts their
share produces a threshold tuned for the wrong distribution.

The alternative to sampling is not practical here. At 35 candidates per
entity the full 2,206,822 S1 training entities expand to ~77M pairs, a
16.4 GB feature matrix, which does not fit. For reference, the project that
reports 0.9545 on the public leaderboard trains on 30% of Source 2/3, and
several others train on 2,000-15,000 S1 entities.

Usage:
    python train_model.py --sample-frac 0.25
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import polars as pl

from code.business_entity_resolution.src.config import (
    F_BETA, MODELS_DIR, RANDOM_SEED, TRAIN_VAL_SPLIT_RATIO,
)
from code.business_entity_resolution.src.data_loader import DataLoader
from code.business_entity_resolution.src.blocking import MultiChannelBlocker
from code.business_entity_resolution.src.features import (
    extract_features_from_records,
)
from code.business_entity_resolution.src.model import TriEnsembleModel
from code.business_entity_resolution.src.decision_rule import (
    DecisionRuleOptimizer, EntityIndex,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-frac", type=float, default=0.25,
                    help="Fraction of non-singleton S1 entities to keep")
    ap.add_argument("--val-sample", action="store_true",
                    help="Train on the small val_sample split instead of full train")
    ap.add_argument("--out", type=Path, default=MODELS_DIR)
    ap.add_argument("--max-candidates", type=int, default=35)
    args = ap.parse_args()

    t_start = time.time()
    loader = DataLoader(use_val_sample=args.val_sample, verbose=True)
    s1, s2, s3, gt = loader.load_data()
    gt_dict = loader.build_ground_truth_dict()

    # ---- stratified sample ------------------------------------------------
    n_total = len(gt_dict)
    sing = [k for k, v in gt_dict.items() if not v]
    non = [k for k, v in gt_dict.items() if v]
    full_singleton_rate = len(sing) / n_total
    log(f"full S1: {n_total:,}  singletons {len(sing):,} "
        f"({full_singleton_rate * 100:.2f}%)  non-singleton {len(non):,}")

    rng = np.random.default_rng(RANDOM_SEED)
    if args.sample_frac >= 1.0:
        keep = list(gt_dict.keys())
    else:
        # Keep every singleton (they are only 5.6% and they matter most),
        # subsample the non-singletons, then restore the singleton rate by
        # trimming the non-singletons to the same ratio.
        n_keep_non = int(len(non) * args.sample_frac)
        # match singleton share of the final sample to the full data
        target_total = len(sing) + n_keep_non
        n_sing = min(len(sing), int(round(target_total * full_singleton_rate)))
        keep_sing = list(rng.choice(sing, size=n_sing, replace=False)) if n_sing < len(sing) else sing
        keep_non = list(rng.choice(non, size=n_keep_non, replace=False))
        keep = keep_sing + keep_non
    keep_set = set(keep)
    rate = sum(1 for k in keep if not gt_dict[k]) / len(keep)
    log(f"sampled S1: {len(keep):,}  singleton rate {rate * 100:.2f}% "
        f"(target {full_singleton_rate * 100:.2f}%)")
    assert abs(rate - full_singleton_rate) < 0.02, "singleton rate drifted"

    s1_sub = s1[s1["entity_id"].isin(keep_set)]
    log(f"S1 frame: {s1_sub.shape[0]:,}")

    # ---- blocking ---------------------------------------------------------
    s1_records = {r["entity_id"]: r for r in s1_sub.to_dict("records")}
    s2_records = {r["entity_id"]: r for r in s2.to_dict("records")}
    s3_records = {r["entity_id"]: r for r in s3.to_dict("records")}
    s23 = {**s2_records, **s3_records}
    log(f"pool: {len(s23):,}")

    blocker = MultiChannelBlocker(verbose=True)
    t0 = time.time()
    candidates = blocker.generate_all_candidates(s1_records, s23)
    n_pairs = sum(len(v) for v in candidates.values())
    log(f"candidates: {n_pairs:,} ({n_pairs / max(len(s1_records), 1):.1f}/S1) "
        f"in {time.time() - t0:.0f}s")

    # ---- features ---------------------------------------------------------
    t0 = time.time()
    feats = extract_features_from_records(
        candidates, s1_records, s23, phase=2, verbose=True)
    labels = [
        1 if cid in gt_dict.get(sid, set()) else 0
        for sid, cid in zip(feats["s1_id"], feats["s2_s3_id"])
    ]
    feats["label"] = labels
    log(f"features: {feats.shape[0]:,} x {feats.shape[1]} in {time.time() - t0:.0f}s")
    log(f"  positives: {sum(labels):,} ({100 * sum(labels) / len(labels):.2f}%)")

    # ---- train ------------------------------------------------------------
    X = feats.drop(columns=["label", "s1_id", "s2_s3_id"], errors="ignore")
    y = feats["label"].to_numpy()
    # group by S1 entity so no entity contributes to two folds
    groups = pd.factorize(feats["s1_id"])[0]

    model = TriEnsembleModel(verbose=True)
    t0 = time.time()
    trained, oof = model.train_groupkfold(X, y, groups)
    log(f"trained in {time.time() - t0:.0f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_models(trained, path=args.out)
    log(f"models saved to {args.out}")

    # ---- threshold + decision rule on OOF ---------------------------------
    import pickle
    from code.business_entity_resolution.src.pairing import (
        build_probabilities_by_s1,
    )
    # Do NOT reset row_index: oof is in fold order, and row_index is the only
    # thing that links an OOF row back to its feature row.
    probabilities_by_s1 = build_probabilities_by_s1(
        oof, feats["s1_id"].to_numpy(), feats["s2_s3_id"].to_numpy())
    log(f"grouped probabilities for {len(probabilities_by_s1):,} S1 entities")

    opt = DecisionRuleOptimizer(beta=F_BETA)
    outcome = opt.grid_search(probabilities_by_s1,
                              {k: v for k, v in gt_dict.items() if k in keep_set},
                              verbose=True)
    best = outcome["best"]
    log(f"BEST rule={best['rule']} score={best['score_tau']} "
        f"margin={best['margin_tau']} alpha={best['alpha']} shift={best['shift']} "
        f"macro F0.5={best['macro_f_beta'] * 100:.4f}%")

    payload = {
        "oof": oof,
        "probabilities_by_s1": probabilities_by_s1,
        "gt_dict": {k: v for k, v in gt_dict.items() if k in keep_set},
        "best": best,
    }
    from code.business_entity_resolution.src.config import OOF_CACHE
    OOF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(OOF_CACHE, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"OOF cache -> {OOF_CACHE}")

    log("total elapsed %.1f min" % ((time.time() - t_start) / 60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
