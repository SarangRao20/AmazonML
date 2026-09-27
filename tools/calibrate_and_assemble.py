#!/usr/bin/env python3
"""
calibrate_and_assemble.py

Rebuilds matching_results.tsv from per-country probability files, choosing
one threshold per country.

Why per-country and not one global tau: the score distribution is not
comparable across countries. Measured on the same sweep,

    tau     US mean   France mean
    0.993      2.91         3.49     <- correct for France
    0.976      3.43         4.52     <- correct for the US
    0.975      3.45         4.56

The US is calibrated near 0.976 and France near 0.993, and no single value
serves both. At 0.975 the US is right and France keeps ~32% too many pairs;
at 0.993 France is right and the US comes up ~16% short. France is the
unseen country, so its probabilities sit lower for equivalent evidence and a
shared floor silently leaves it over-committed. The score is a macro average,
so one broken country caps the whole submission -- France left unfixed holds
the macro near 0.83 however good the US and India are.

The target is the training prior, not a fitted constant: US and India in
train both have mean 3.46 matches per S1 and 5.58% singletons -- an identical
distribution, which suggests the generator drew all three countries the same
way. tau is solved per country by bisection so the predicted mean lands on
3.46. The resulting empty rate is reported as a check rather than fitted; it
should land near 5.5% on its own.
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

TARGET_MEAN = 3.46


def load_scores(path):
    """entity_id -> {candidate_id: probability}"""
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("source1"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            d.setdefault(p[0], {})[p[1]] = float(p[2])
    return d


def solve_tau(scores, n_total, target=TARGET_MEAN, lo=0.5, hi=0.9999):
    """Bisect tau so the mean count over ALL S1 in the country hits target.

    n_total counts entities that never reached the score file, i.e. the ones
    the first run left empty. They contribute zero matches but must stay in
    the denominator; dropping them inflates the mean and drives tau far too
    low, which is exactly what produced the 0.712 submission.
    """
    arrays = [np.fromiter(v.values(), dtype=float, count=len(v))
              for v in scores.values()]
    for _ in range(40):
        mid = (lo + hi) / 2
        s = sum(int((a >= mid).sum()) for a in arrays)
        # mean is decreasing in tau: too few matches means tau is too high,
        # so pull the upper bound down; too many means push the lower bound up.
        if s / n_total < target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", default="output/parts")
    ap.add_argument("--matching", default="output/matching_results.tsv")
    ap.add_argument("--out", default="output_calibrated")
    ap.add_argument("--test-dir", default="dataset/test")
    args = ap.parse_args()

    # Row set comes from the existing submission, not the score files: the
    # score files only cover entities that had candidates, and every S1 must
    # appear exactly once.
    order, submitted = [], {}
    with open(args.matching, encoding="utf-8") as f:
        next(f)
        for line in f:
            sid, _, rest = line.rstrip("\n").partition("\t")
            order.append(sid)
            cs = [x for x in rest.split(",") if x]
            if cs:
                submitted[sid] = cs

    import polars as pl
    s1 = pl.read_csv(f"{args.test_dir}/test_source1.tsv", separator="\t",
                     columns=["entity_id", "country"])
    country = {r[0]: r[1] for r in s1.iter_rows()}

    n_by_country = defaultdict(int)
    for sid in order:
        n_by_country[country[sid]] += 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Candidates are unchanged: calibration only removes pairs, and every
    # final match must still appear in candidate_pairs.tsv.
    shutil.copy(f"{Path(args.matching).parent}/candidate_pairs.tsv",
                out / "candidate_pairs.tsv")

    kept_by_country = {}
    for c in sorted(n_by_country):
        sp = Path(args.scores_dir) / f"{c}.scores.tsv"
        if not sp.exists():
            print(f"  {c}: NO scores file -- keeping submitted matches as-is")
            kept_by_country[c] = None
            continue
        raw = load_scores(sp)
        # Restrict to submitted pairs BEFORE solving tau, not after. The
        # bisection counts matches, so feeding it ids we are not allowed to
        # emit would drive tau down and undershoot the target mean.
        sc = {sid: {cid: p for cid, p in cands.items()
                    if cid in set(submitted.get(sid, []))}
              for sid, cands in raw.items()}
        sc = {sid: c for sid, c in sc.items() if c}
        tau = solve_tau(sc, n_by_country[c])
        keep = {sid: [cid for cid, p in cands.items() if p >= tau]
                for sid, cands in sc.items()}
        kept_by_country[c] = (tau, keep)
        kept = sum(len(v) for v in keep.values())
        empty = n_by_country[c] - sum(1 for v in keep.values() if v)
        print(f"  {c}: tau={tau:.5f}  mean={kept / n_by_country[c]:.3f} "
              f"(target {TARGET_MEAN})  empty={100 * empty / n_by_country[c]:.2f}%")

    n_kept = 0
    with open(out / "matching_results.tsv", "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in order:
            c = country[sid]
            entry = kept_by_country.get(c)
            if entry is None or sid not in submitted:
                cs = submitted.get(sid, [])
            else:
                cs = entry[1].get(sid, [])
            f.write(f"{sid}\t{','.join(cs)}\n")
            n_kept += len(cs)

    print(f"  wrote {out/'matching_results.tsv'}: {len(order):,} rows, "
          f"{n_kept:,} pairs, mean {n_kept / len(order):.3f}")


if __name__ == "__main__":
    main()
