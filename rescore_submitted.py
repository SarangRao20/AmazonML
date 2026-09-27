#!/usr/bin/env python3
"""
rescore_submitted.py

Scores only the pairs already present in a submitted matching file, and
writes <entity_id>\\t<candidate_id>\\t<probability>.

Why this exists: the full test run took 9 hours and, because it predates the
score side-file, wrote no probabilities. Re-deriving them the obvious way --
re-running inference -- would cost another 9 hours, which does not fit in the
remaining budget. But the submitted file already contains the pairs we care
about, and the problem is purely that too many of them were kept. Removing
pairs only ever needs scores on pairs we already have, so scoring 7.3M
submitted pairs replaces scoring 68M candidates. That is a ~9x reduction and
is the difference between fitting in the deadline and not.

The ceiling is real and worth stating: because this only ever re-scores pairs
that survived the previous rule, it can lower the predicted count but cannot
raise it. That is the right direction here -- the submitted file over-predicts
by 43% against the train prior, so the fix is subtraction, not addition. If
the calibrated run later shows we are now under-predicting, this script cannot
correct that and a fresh blocking pass would be required.

Reads:  <matching>            a submitted matching_results.tsv
Writes: <out>/<country>.scores.tsv
"""

import argparse
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl

from code.business_entity_resolution.src.features import (
    RecView as _RecView,
    extract_features_from_records,
)
from code.business_entity_resolution.src.model import TriEnsembleModel

S1_CHUNK = 3_000


def build_records(df):
    """Rows -> the {entity_id: (business_name, business_address)} mapping that
    RecView wraps. This shape is load-bearing: passing a dict of parallel lists
    instead makes the ratio features come back as object dtype, and xgboost
    rejects the frame outright."""
    return {r["entity_id"]: (r["business_name"], r["business_address"])
            for r in df.iter_rows(named=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", default="output/matching_results.tsv")
    ap.add_argument("--out", default="output/parts")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--country", action="append", default=None,
                    help="restrict to these countries")
    args = ap.parse_args()

    from run_test_inference import load_test

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- what did we submit -------------------------------------------------
    want = defaultdict(list)
    with open(args.matching, encoding="utf-8") as f:
        next(f)
        for line in f:
            sid, _, rest = line.rstrip("\n").partition("\t")
            cs = [x for x in rest.split(",") if x]
            if cs:
                want[sid] = cs
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    log(f"submitted pairs to score: {sum(len(v) for v in want.values()):,} "
        f"across {len(want):,} entities")

    s1_all, s2, s3 = load_test()
    s1_map = {r[0]: r for r in
              s1_all.select(["entity_id", "country"]).iter_rows()}
    log("models loading")
    model = TriEnsembleModel(verbose=True)
    models = model.load_models(path=args.model_dir)
    log("  models loaded")

    pool = pl.concat([s2, s3], how="vertical_relaxed")
    pool = pool.with_columns(
        pl.col("entity_id").cast(pl.Utf8))

    by_country = defaultdict(list)
    for sid in want:
        r = s1_map.get(sid)
        if r is None:
            continue
        by_country[r[1]].append(sid)
    log(f"by country: { {k: len(v) for k, v in by_country.items()} }")

    countries = args.country or sorted(by_country)
    for country in countries:
        sids = by_country.get(country, [])
        if not sids:
            log(f"{country}: nothing submitted, skipping")
            continue

        a = s1_all.filter(pl.col("country") == country)
        b = pool.filter(pl.col("country") == country)
        s1_recs = build_records(a)
        s23_recs = build_records(b)
        s1_view = _RecView(s1_recs)
        s23_view = _RecView(s23_recs)
        keep = set(sids)
        t0 = time.time()
        n = 0
        with open(out_dir / f"{country}.scores.tsv", "w", encoding="utf-8") as out:
            out.write("source1_entity_id\tcandidate_entity_id\tprobability\n")
            for start in range(0, len(sids), S1_CHUNK):
                batch = [s for s in sids[start:start + S1_CHUNK] if s in keep]
                if not batch:
                    continue
                # The blocker score is unpacked and then discarded (features.py
                # only keeps the ids, then recomputes similarity from the
                # strings), so a dummy is faithful here rather than a shortcut.
                chunk = {s: [(c, 0.0) for c in want[s]] for s in batch}
                feats = extract_features_from_records(
                    chunk, s1_view, s23_view, phase=2, verbose=False)
                X = feats.drop(columns=["s1_id", "s2_s3_id"], errors="ignore")
                probs = model.predict_ensemble(X, models, use_calibration=True)
                rows = zip(feats["s1_id"], feats["s2_s3_id"], probs)
                for sid, cid, p in sorted(rows, key=lambda t: (t[0], -t[2])):
                    out.write(f"{sid}\t{cid}\t{float(p):.6f}\n")
                n += len(batch)
                del feats, X, probs
                gc.collect()
                if n % 30_000 < S1_CHUNK:
                    log(f"  {country}: {n:,}/{len(sids):,} "
                        f"({time.time() - t0:.0f}s)")
        log(f"{country}: wrote scores for {n:,} entities in "
            f"{time.time() - t0:.0f}s")
        del a, b, s1_recs, s23_recs, s1_view, s23_view
        gc.collect()


if __name__ == "__main__":
    main()
