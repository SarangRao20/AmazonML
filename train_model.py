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
import gc
import os
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
from code.business_entity_resolution.src.blocking_sparse import SparseBlocker
from code.business_entity_resolution.src.features import (
    RecView, extract_features_from_records, fixed_chunks,
)
from code.business_entity_resolution.src.model import TriEnsembleModel
from code.business_entity_resolution.src.decision_rule import (
    DecisionRuleOptimizer, EntityIndex,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _iter_records(frame):
    """Stream (country, entity_id, name, address) from a pandas frame.

    Not to_dict("records"): on the 5.03M/5.29M-row pool frames that
    materialises a list of five million dicts, roughly 2 GB of headers,
    before the first blocking shard is built. to_numpy() on an object
    column yields an array of references to the existing strings, so the
    four columns together cost ~160 MB.
    """
    return zip(frame["country"].to_numpy(),
               frame["entity_id"].to_numpy(),
               frame["business_name"].to_numpy(),
               frame["business_address"].to_numpy())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-frac", type=float, default=0.25,
                    help="Fraction of non-singleton S1 entities to keep")
    ap.add_argument("--val-sample", action="store_true",
                    help="Train on the small val_sample split instead of full train")
    ap.add_argument("--out", type=Path, default=MODELS_DIR)
    ap.add_argument("--compare-models", type=Path, default=None,
                    help="Score the incumbent model dir on this run's OOF rows")
    ap.add_argument("--verdict-file", type=Path,
                    default=Path("logs/retrain_verdict.txt"),
                    help="Where to write NEW WINS / INCUMBENT WINS for the runner")
    ap.add_argument("--max-candidates", type=int, default=45,
                    help="recall ceiling; 45 reaches 98.4%% pair recall "
                         "on val_sample vs 95.2%% at 35")
    ap.add_argument("--blocker", choices=["sparse", "dict"],
                    default="sparse")
    args = ap.parse_args()
    FEAT_CHUNK = int(os.environ.get("ER_TRAIN_CHUNK", 20_000))

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
        # Subsample BOTH strata at the same rate. Keeping every singleton
        # while subsampling only non-singletons silently over-represents
        # them, and singletons are the easy negatives: at frac 0.01 that
        # put 27.8% singletons in the sample against a 5.58% target, which
        # shifts the prior the model calibrates its thresholds against.
        n_keep_sing = int(round(len(sing) * args.sample_frac))
        n_keep_non = int(round(len(non) * args.sample_frac))
        keep_sing = (list(rng.choice(sing, size=n_keep_sing, replace=False))
                     if n_keep_sing < len(sing) else sing)
        keep_non = (list(rng.choice(non, size=n_keep_non, replace=False))
                    if n_keep_non < len(non) else non)
        keep = keep_sing + keep_non
    keep_set = set(keep)
    rate = sum(1 for k in keep if not gt_dict[k]) / len(keep)
    log(f"sampled S1: {len(keep):,}  singleton rate {rate * 100:.2f}% "
        f"(target {full_singleton_rate * 100:.2f}%)")
    assert abs(rate - full_singleton_rate) < 0.02, "singleton rate drifted"

    s1_sub = s1[s1["entity_id"].isin(keep_set)]
    log(f"S1 frame: {s1_sub.shape[0]:,}")

    # ---- blocking, per country and streamed -------------------------------
    # Country partitioning is lossless: not one of the 7,638,365 ground-truth
    # pairs crosses a country boundary, and it bounds peak memory at the
    # largest single country instead of the whole 10.3M pool.
    #
    # Records are (name, address) tuples, not dicts. At 10.3M pool records
    # the 4-key dict headers alone are ~4 GB.
    #
    # Blocking against the FULL per-country pool is deliberate: the 18
    # channel/rank/density features are only calibrated if the candidate
    # density during training matches test. val_sample's 518k pool is an
    # order of magnitude smaller than India's 4.72M test pool, which left
    # those features shifted.
    s1_by_country: dict = {}
    for country, eid, nm, ad in _iter_records(s1_sub):
        s1_by_country.setdefault(country, {})[eid] = (nm, ad)
    pool_by_country: dict = {}
    for frame in (s2, s3):
        for country, eid, nm, ad in _iter_records(frame):
            pool_by_country.setdefault(country, {})[eid] = (nm, ad)
    log(f"pool: {sum(len(v) for v in pool_by_country.values()):,} across "
        f"{sorted(pool_by_country)}")

    blocker = SparseBlocker(verbose=True) if args.blocker == "sparse" else None
    feat_parts = []
    n_pairs = 0
    tot_true = tot_captured = n_non_single = n_full = 0
    t0 = time.time()
    for country, s1_recs in sorted(s1_by_country.items()):
        pool_recs = pool_by_country.get(country) or {}
        log(f"  {country}: S1={len(s1_recs):,} pool={len(pool_recs):,}")
        if not pool_recs:
            log(f"    no pool records for {country}; "
                f"{len(s1_recs):,} S1 entities get no candidates")
            continue
        if blocker is not None:
            stream = blocker.iter_candidate_chunks(
                s1_recs, pool_recs, max_candidates=args.max_candidates)
        else:
            cands = MultiChannelBlocker(
                verbose=True).generate_all_candidates(s1_recs, pool_recs)
            stream = fixed_chunks(cands, FEAT_CHUNK)
        s1_view, pool_view = RecView(s1_recs), RecView(pool_recs)
        for ids, chunk in stream:
            feats_part = extract_features_from_records(
                chunk, s1_view, pool_view, phase=2, verbose=False)
            # Cast per chunk, not after the concat: at ~8M pairs the parts
            # list plus a float64 concat peaks around 8 GB. Casting as each
            # chunk lands keeps the whole set near 1.8 GB.
            feats_part = feats_part.astype(
                {c: np.float32 for c in feats_part.columns
                 if c not in ("s1_id", "s2_s3_id")})
            feat_parts.append(feats_part)
            n_pairs += sum(len(v) for v in chunk.values())
            for sid in ids:
                truth = gt_dict.get(sid, set())
                got = {c for c, _ in chunk.get(sid, [])}
                tot_true += len(truth)
                tot_captured += len(truth & got)
                if truth:
                    n_non_single += 1
                    if truth.issubset(got):
                        n_full += 1
            del feats_part, chunk
            gc.collect()
        del s1_recs, pool_recs, s1_view, pool_view
        gc.collect()
    log(f"candidates: {n_pairs:,} ({n_pairs / max(len(s1_sub), 1):.1f}/S1) "
        f"in {time.time() - t0:.0f}s")
    log(f"BLOCKING pair recall={tot_captured / max(tot_true, 1) * 100:.2f}%  "
        f"entity full coverage={n_full / max(n_non_single, 1) * 100:.2f}%")

    # ---- features ---------------------------------------------------------
    t0 = time.time()
    feats = pd.concat(feat_parts, ignore_index=True)
    del feat_parts
    gc.collect()
    # float32 halves both the frame and the DMatrix the boosters build from
    # it; 8M+ pairs is where that difference stops being noise.
    feats = feats.astype({c: np.float32 for c in feats.columns
                          if c not in ("s1_id", "s2_s3_id")})
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

    # ---- incumbent comparison on identical rows ---------------------------
    # Both models are scored on this run's OOF feature matrix and each gets
    # its own best decision rule, so neither is handed a threshold tuned for
    # the other. The comparison is deliberately conservative: the new
    # model's numbers here are out-of-fold, while the incumbent was trained
    # on a superset that includes some of these entities, so its score is
    # flattered by in-sample exposure. New >= old despite that handicap is
    # strong evidence, not a coin flip.
    if args.compare_models:
        old_dir = Path(args.compare_models)
        if (old_dir / "xgboost_model.json").exists() or list(old_dir.glob("*")):
            try:
                old_model = TriEnsembleModel(verbose=False)
                old_loaded = old_model.load_models(path=old_dir)
                old_probs = old_model.predict_ensemble(
                    X, old_loaded, use_calibration=True)
                old_by_s1 = build_probabilities_by_s1(
                    old_probs, feats["s1_id"].to_numpy(),
                    feats["s2_s3_id"].to_numpy())
                old_out = DecisionRuleOptimizer(beta=F_BETA).grid_search(
                    old_by_s1, {k: v for k, v in gt_dict.items()
                                if k in keep_set}, verbose=False)
                log(f"INCUMBENT ({old_dir}) macro F0.5="
                    f"{old_out['best']['macro_f_beta'] * 100:.4f}%  "
                    f"rule={old_out['best']['rule']} "
                    f"score={old_out['best']['score_tau']} "
                    f"alpha={old_out['best']['alpha']}")
                log(f"NEW model          macro F0.5="
                    f"{best['macro_f_beta'] * 100:.4f}%  "
                    f"rule={best['rule']} score={best['score_tau']} "
                    f"alpha={best['alpha']}")
                verdict = ("NEW WINS" if best["macro_f_beta"]
                           >= old_out["best"]["macro_f_beta"] else "INCUMBENT WINS")
                log(f"VERDICT: {verdict} (margin "
                    f"{(best['macro_f_beta'] - old_out['best']['macro_f_beta']) * 100:+.4f} pts)")
                with open(Path(args.verdict_file), "w", encoding="utf-8") as fh:
                    fh.write(f"{verdict}\n")
                    fh.write(f"new={best['macro_f_beta']:.6f}\n")
                    fh.write(f"incumbent={old_out['best']['macro_f_beta']:.6f}\n")
            except Exception as exc:  # comparison must never kill the run
                log(f"incumbent comparison failed (non-fatal): "
                    f"{type(exc).__name__}: {exc}")
                xcols = list(X.columns) if hasattr(X, "columns") else []
                log(f"  diag: X={type(X).__name__} feats={type(feats).__name__} "
                    f"n_x_cols={len(xcols)} first5={xcols[:5]}")
        else:
            log(f"no incumbent models in {old_dir}, skipping comparison")

    # Persist the winning rule beside the models so inference adopts the
    # thresholds the model was actually calibrated against, instead of
    # relying on run_test_inference's defaults still matching.
    import json
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "decision_rule.json", "w", encoding="utf-8") as fh:
        json.dump({k: (float(v) if isinstance(v, (int, float)) else v)
                   for k, v in best.items()}, fh, indent=2)
    log(f"decision rule -> {args.out / 'decision_rule.json'}")

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
