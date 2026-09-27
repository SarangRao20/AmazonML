#!/usr/bin/env python3
"""
decide_entity.py

Per-S1-entity decision rule that directly maximises expected F0.5, instead of
applying one threshold to every entity.

Why this replaces a global threshold. A threshold answers "is this pair a
match" for each pair independently, but the metric is scored per entity and
depends on the *count* chosen: an entity with four mediocre candidates and
one with four near-certain ones should not be treated alike, and neither
should either of them if all four candidates are weak enough that the whole
entity is probably a singleton. Reference project
`reference_projects_9` reports 0.9545 with this rule, against 0.9545-level
runs that tune only a threshold, so the shape of the decision is the part
that matters.

The rule, for one entity with candidates whose probabilities are q_1..q_n
after recalibration:

    keep none :  E[F] = prod_i (1 - q_i)      (probability it is a singleton)
    keep k    :  E[F] = 1.25 * (sum of top-k q) / (0.25 * sum of all q + k)

Pick the k with the highest E[F], and empty the entity if "none" beats it.
q is a sigmoid of the logit shifted by `shift`, which is the single knob
tuned out of fold; it exists because the ensemble is over-confident and a
raw probability is not a calibrated frequency.

This runs on the probability side-files, so it is post-processing only and
needs no retraining.
"""

import argparse
from collections import defaultdict

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def entity_decision(probs, shift):
    """Return the number of candidates to keep for one entity.

    probs must be sorted descending.
    """
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    q = sigmoid(np.log(p / (1 - p)) + shift)

    # E[F] for keeping nothing: the entity is a singleton only if every
    # candidate is wrong.
    p_none = float(np.prod(1.0 - q))

    n = len(q)
    if n == 0:
        return 0, p_none, 0.0
    q_all = float(q.sum())
    cum = np.cumsum(q)
    ks = np.arange(1, n + 1)
    ef = 1.25 * cum / (0.25 * q_all + ks)

    k_best = int(np.argmax(ef)) + 1
    ef_best = float(ef.max())
    if p_none >= ef_best:
        return 0, p_none, ef_best
    return k_best, p_none, ef_best


def apply_rule(scores_by_s1, shift, max_per_s1=11):
    """scores_by_s1: {s1_id: [(candidate_id, prob), ...]} -> {s1_id: [ids]}"""
    out = {}
    for sid, cands in scores_by_s1.items():
        if not cands:
            out[sid] = []
            continue
        ordered = sorted(cands, key=lambda t: -t[1])[:max_per_s1]
        k, _, _ = entity_decision([p for _, p in ordered], shift)
        out[sid] = [cid for cid, _ in ordered[:k]]
    return out


def macro_f05(pred, truth):
    tot = 0.0
    for sid, t in truth.items():
        t = set(t)
        p = set(pred.get(sid, []))
        if not t and not p:
            tot += 1.0
        elif t or p:
            tot += 5 * len(t & p) / (len(t) + 4 * len(p))
    return tot / len(truth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tune", "apply"], default="tune")
    ap.add_argument("--scores-dir", default="output/parts")
    ap.add_argument("--matching", default="output/matching_results.tsv")
    ap.add_argument("--out", default="output_entity")
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--shift", type=float, default=None)
    args = ap.parse_args()

    if args.mode == "tune":
        import pickle
        d = pickle.load(open("models/oof_predictions.pkl", "rb"))
        P = {k: v for k, v in d["probabilities_by_s1"].items()}
        G = d["gt_dict"]
        print(f"  OOF entities: {len(P):,}")
        base = {s: [c for c, p in cs if p >= 0.65] for s, cs in P.items()}
        print(f"  baseline (tau=0.65): F0.5 = {macro_f05(base, G):.4f}")
        best = (0, None)
        for shift in np.arange(-6, 6.01, 0.25):
            pred = apply_rule(P, float(shift))
            f = macro_f05(pred, G)
            if f > best[0]:
                best = (f, float(shift))
            if abs(shift % 1.0) < 1e-9:
                print(f"    shift={shift:+.2f}  F0.5={f:.4f}  "
                      f"mean={np.mean([len(v) for v in pred.values()]):.2f}")
        print(f"  BEST shift={best[1]:+.2f}  F0.5={best[0]:.4f}")
        return

    from pathlib import Path
    import polars as pl
    s1 = pl.read_csv(f"{args.test_dir}/test_source1.tsv", separator="\t",
                     columns=["entity_id", "country"])
    country = {r[0]: r[1] for r in s1.iter_rows()}

    order = []
    with open(args.matching, encoding="utf-8") as f:
        next(f)
        for line in f:
            order.append(line.rstrip("\n").partition("\t")[0])

    by_country = defaultdict(list)
    for sid in order:
        by_country[country[sid]].append(sid)

    shift = args.shift
    if shift is None:
        shift = float(input("shift: "))

    result = {sid: [] for sid in order}
    for c, sids in sorted(by_country.items()):
        sp = Path(args.scores_dir) / f"{c}.scores.tsv"
        if not sp.exists():
            print(f"  {c}: no scores, left empty")
            continue
        scores = defaultdict(list)
        with open(sp, encoding="utf-8") as f:
            for line in f:
                if line.startswith("source1"):
                    continue
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3:
                    scores[p[0]].append((p[1], float(p[2])))
        got = apply_rule(scores, shift)
        n = 0
        for sid in sids:
            cs = got.get(sid, [])
            result[sid] = cs
            n += len(cs)
        emp = sum(1 for sid in sids if not result[sid])
        print(f"  {c}: mean={n/len(sids):.3f}  empty={100*emp/len(sids):.2f}%  "
              f"({len(sids):,} entities)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(f"{Path(args.matching).parent}/candidate_pairs.tsv",
                out / "candidate_pairs.tsv")
    with open(out / "matching_results.tsv", "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in order:
            f.write(f"{sid}\t{','.join(result[sid])}\n")
    tot = sum(len(v) for v in result.values())
    print(f"  wrote {out/'matching_results.tsv'}: {len(order):,} rows, "
          f"{tot:,} pairs, mean {tot/len(order):.3f}")


if __name__ == "__main__":
    main()
