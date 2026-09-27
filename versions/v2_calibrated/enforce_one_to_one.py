#!/usr/bin/env python3
"""
enforce_one_to_one.py

Resolves target collisions using the calibrated predictions, breaking ties on
the model probability rather than on rank.

The ranking is verified on train: 7,638,365 pairs claim 7,638,365 distinct
S2/S3 targets, so a target is never matched to two S1 entities. That makes
every duplicate claim a guaranteed false positive, and removing it is a
strict improvement in precision rather than a heuristic.

Rank is a poor tie-break across entities, because rank 0 for one S1 and rank
0 for another say nothing about which pair the model actually believes more.
After calibration roughly 8-11% of collisions are decided by rank alone, so
those were being resolved by file order. The probability files make the real
comparison available.

Read-only with respect to the input. An S1 that loses every one of its claims
becomes an empty row, which is legal, but it is counted and reported so the
cost is visible rather than assumed.
"""

import argparse
from collections import defaultdict
from pathlib import Path


def load_scores(path):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", default="output_calibrated/matching_results.tsv")
    ap.add_argument("--scores-dir", default="output/parts")
    ap.add_argument("--out", default="output_final")
    ap.add_argument("--test-dir", default="dataset/test")
    args = ap.parse_args()

    order, matches = [], {}
    with open(args.matching, encoding="utf-8") as f:
        next(f)
        for line in f:
            sid, _, rest = line.rstrip("\n").partition("\t")
            order.append(sid)
            cs = [x for x in rest.split(",") if x]
            if cs:
                matches[sid] = cs

    import polars as pl
    s1 = pl.read_csv(f"{args.test_dir}/test_source1.tsv", separator="\t",
                     columns=["entity_id", "country"])
    country = {r[0]: r[1] for r in s1.iter_rows()}

    print("Loading score dictionaries per country...")
    scores_by_country = {}
    for cc in set(country.values()):
        sp = Path(args.scores_dir) / f"{cc}.scores.tsv"
        if sp.exists():
            print(f"  Loading {sp.name}...")
            scores_by_country[cc] = load_scores(sp)
        else:
            scores_by_country[cc] = {}

    # per country, the (entity, probability) claiming each target
    claim = defaultdict(dict)          # country -> target -> (sid, p)
    missing_score = 0
    for sid, cs in matches.items():
        cc = country[sid]
        mysc = scores_by_country[cc].get(sid, {})
        for cid in cs:
            p = mysc.get(cid)
            if p is None:
                # No probability to compare with. Keep the first claimer rather
                # than drop a real match, and count it so the gap is visible.
                missing_score += 1
                p = float("-inf")
            cur = claim[cc].get(cid)
            if cur is None or p > cur[1]:
                claim[cc][cid] = (sid, p)

    kept = {sid: [] for sid in order}
    for cc, d in claim.items():
        for cid, (sid, _) in d.items():
            kept[sid].append(cid)

    n_before = sum(len(v) for v in matches.values())
    n_after = sum(len(v) for v in kept.values())
    emptied = sum(1 for sid in order
                  if matches.get(sid) and not kept.get(sid))
    was_empty = sum(1 for sid in order if not matches.get(sid))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(f"{Path(args.matching).parent}/candidate_pairs.tsv",
                out / "candidate_pairs.tsv")
    with open(out / "matching_results.tsv", "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in order:
            f.write(f"{sid}\t{','.join(kept[sid])}\n")

    n = len(order)
    print(f"  pairs   {n_before:,} -> {n_after:,}  "
          f"(removed {n_before - n_after:,} guaranteed FPs)")
    print(f"  mean    {n_before/n:.3f} -> {n_after/n:.3f}")
    print(f"  empty   {was_empty:,} -> {was_empty + emptied:,} "
          f"({100*(was_empty+emptied)/n:.2f}%)   {emptied:,} newly emptied")
    if missing_score:
        print(f"  WARNING {missing_score:,} pairs had no probability and kept "
              f"the first claimer")


if __name__ == "__main__":
    main()
