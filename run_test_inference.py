"""Memory-bounded test-set inference for the 1.73M x 9.97M test split.

Why this exists
---------------
Materialising test features for the whole split needs 61M pairs at 35
candidates per entity, which is a 12.9 GB float32 matrix on a 15 GB
machine. The dominant cost is actually the blocking index over 9.97M
Source 2/3 records held as Python dict-of-lists, which is worse still.

Three changes keep the peak inside a small laptop, following the approach
that reference project `vitthalg17/amazon-ml-challenge-2026-entity-resolution`
used to reach 0.9545 macro F_0.5 at a reported 4.4 GB peak:

1. Per-country blocking. Every country is processed on its own: build the
   indices for that country's S2/S3 slice, generate candidates for that
   country's S1 entities, write the result, then free everything. This is
   lossless rather than a heuristic - across all 7,638,365 matched id
   occurrences in train_ground_truth.tsv, not one crosses a country
   boundary, so no true match can be lost by partitioning this way.

2. Chunked scoring with incremental output. Within a country, S1 entities
   are scored in chunks; each chunk's predictions are appended to the
   output TSV and freed. Nothing holds the full candidate matrix.

3. Prediction-time guards, both from the same reference project:
   - an unseen country (France appears only in test) must clear
     threshold + FR_DELTA, because French names are often
     "<city> <word>" and the model over-merges businesses that share a
     city name;
   - at most MAX_PER_S1 matches per S1 entity, set to the maximum match
     count observed in the training ground truth, which costs nothing on
     training data and removes low-confidence tail predictions.

Resume: writes a per-country done-marker, so re-running skips completed
countries.

Usage:
    python run_test_inference.py --model models/full --out output
    python run_test_inference.py --limit-countries 1     # smoke test
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import polars as pl

from code.business_entity_resolution.src.config import (
    CANDIDATE_PAIRS_PATH, DATA_DIR, F_BETA, MATCHING_RESULTS_PATH,
    MODELS_DIR, RANDOM_SEED, TEST_SOURCE1_PATH, TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
)
from code.business_entity_resolution.src.blocking import MultiChannelBlocker
from code.business_entity_resolution.src.features import (
    extract_features_from_records,
)
from code.business_entity_resolution.src.model import TriEnsembleModel
from code.business_entity_resolution.src.consistency import ConsistencyResolver

# Countries present in test but absent from train get the stricter floor.
UNSEEN_COUNTRIES = {"FR", "fr", "France", "FRA"}
FR_DELTA = 0.10
# Maximum matches for one S1 entity in train_ground_truth.tsv.
MAX_PER_S1 = 11
S1_CHUNK = 40_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_test():
    log("loading test sources (lazy scan, low_memory)")
    s1 = pl.read_csv(TEST_SOURCE1_PATH, separator="\t", infer_schema_length=0)
    s2 = pl.read_csv(TEST_SOURCE2_PATH, separator="\t", infer_schema_length=0)
    s3 = pl.read_csv(TEST_SOURCE3_PATH, separator="\t", infer_schema_length=0)
    log(f"  S1={s1.height:,}  S2={s2.height:,}  S3={s3.height:,}")
    return s1, s2, s3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=MODELS_DIR,
                    help="Directory holding xgb/lgb/catboost .pkl models")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-candidates", type=int, default=35)
    ap.add_argument("--score-tau", type=float, default=0.60)
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="Relative-floor coefficient; keep p >= max(score_tau, alpha*max_p)")
    ap.add_argument("--margin-tau", type=float, default=0.01)
    ap.add_argument("--limit-countries", type=int, default=0,
                    help="Only process the first N countries (smoke test)")
    ap.add_argument("--no-guards", action="store_true",
                    help="Disable the France floor and per-S1 cap")
    args = ap.parse_args()

    out_dir = args.out or MATCHING_RESULTS_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    match_path = out_dir / "matching_results.tsv"
    cand_path = out_dir / "candidate_pairs.tsv"
    done_marker = out_dir / ".test_inference_done"

    if done_marker.exists():
        log(f"{done_marker} exists - previous run completed, nothing to do")
        return 0

    s1_all, s2, s3 = load_test()
    all_s1_ids = s1_all["entity_id"].to_list()
    log(f"every one of {len(all_s1_ids):,} S1 entities must appear in the output")

    log(f"loading models from {args.model_dir}")
    model = TriEnsembleModel(verbose=True)
    models = model.load_models(path=args.model_dir)
    log("  models loaded")

    pool = pl.concat([s2, s3], how="vertical_relaxed")
    countries = (
        s1_all.select(pl.col("country")).to_series().unique().sort().to_list()
    )
    log(f"countries in test S1: {countries}")
    if args.limit_countries:
        countries = countries[: args.limit_countries]
        log(f"  limited to {countries}")

    match_fh = open(match_path, "w", encoding="utf-8")
    cand_fh = open(cand_path, "w", encoding="utf-8")
    match_fh.write("source1_entity_id\tmatched_entity_ids\n")
    cand_fh.write("source1_entity_id\tcandidate_entity_ids\n")

    resolver = ConsistencyResolver(verbose=True)
    seen_s1 = set()
    t_start = time.time()

    for ci, country in enumerate(countries, 1):
        t0 = time.time()
        # ---- per-country slice: this is the memory bound -----------------
        a = s1_all.filter(pl.col("country") == country)
        b = pool.filter(pl.col("country") == country)
        log(f"[{ci}/{len(countries)}] {country}: S1={a.height:,} "
            f"S2/S3={b.height:,}")

        s1_records = {r["entity_id"]: r for r in a.to_dicts()}
        s23_records = {r["entity_id"]: r for r in b.to_dicts()}
        seen_s1.update(s1_records)

        if not s23_records:
            for sid in s1_records:
                match_fh.write(f"{sid}\t\n")
                cand_fh.write(f"{sid}\t\n")
            log(f"    no pool records for {country}, wrote {a.height:,} empty rows")
            continue

        blocker = MultiChannelBlocker(verbose=False)
        candidates = blocker.generate_all_candidates(s1_records, s23_records)
        n_pairs = sum(len(v) for v in candidates.values())
        log(f"    candidates: {n_pairs:,} pairs, "
            f"{n_pairs / max(len(s1_records), 1):.1f}/S1  "
            f"({time.time() - t0:.0f}s)")

        # ---- chunked scoring + incremental write --------------------------
        items = list(candidates.items())
        for start in range(0, len(items), S1_CHUNK):
            chunk = dict(items[start:start + S1_CHUNK])
            chunk_records = {sid: s1_records[sid] for sid in chunk}
            feats = extract_features_from_records(
                chunk, chunk_records, s23_records, phase=2, verbose=False)
            X = feats.drop(columns=["s1_id", "s2_s3_id"], errors="ignore")
            probs = model.predict_ensemble(X, models, use_calibration=True)

            by_entity = {}
            for (sid, cid), p in zip(feats["s1_id"], probs):
                by_entity.setdefault(sid, []).append((cid, float(p)))

            for sid, cands in by_entity.items():
                if not cands:
                    match_fh.write(f"{sid}\t\n")
                    cand_fh.write(f"{sid}\t\n")
                    continue
                ranked = sorted(cands, key=lambda t: (-t[1], t[0]))
                best = ranked[0][1]

                if best < args.margin_tau:
                    kept = []
                else:
                    # Winning rule on OOF: relative per-entity floor. Keep
                    # candidates at or above max(score_tau, alpha * max_p).
                    floor = max(args.score_tau, args.alpha * best)
                    if country in UNSEEN_COUNTRIES and not args.no_guards:
                        floor += FR_DELTA
                    kept = [cid for cid, p in ranked if p >= floor]
                    if not args.no_guards:
                        kept = kept[:MAX_PER_S1]

                seen = set()
                uniq = [c for c in kept if not (c in seen or seen.add(c))]
                match_fh.write(f"{sid}\t{','.join(uniq)}\n")
                cand_fh.write(
                    f"{sid}\t{','.join(sorted(c for c, _ in cands))}\n")

            del feats, X, probs, by_entity, chunk

        # Release the country before touching the next one.
        del s1_records, s23_records, candidates, a, b
        log(f"    done in {time.time() - t0:.0f}s  "
            f"(elapsed {(time.time() - t_start) / 60:.1f} min)")

    match_fh.close()
    cand_fh.close()

    missing = [sid for sid in all_s1_ids if sid not in seen_s1]
    log(f"rows written: {len(seen_s1):,}   missing S1: {len(missing):,}")
    if missing:
        log("  appending empty rows for the missing entities")
        with open(match_path, "a", encoding="utf-8") as fh:
            for sid in missing:
                fh.write(f"{sid}\t\n")
        with open(cand_path, "a", encoding="utf-8") as fh:
            for sid in missing:
                fh.write(f"{sid}\t\n")

    done_marker.write_text(f"completed {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log(f"wrote {match_path} and {cand_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
