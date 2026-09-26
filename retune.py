"""Re-apply a decision rule to saved candidate scores, without re-scoring.

Why this exists
---------------
``run_test_inference`` used to fuse scoring and thresholding together, so
changing the decision rule meant re-running inference: 37 minutes for
France, 4.7 hours for the whole test set. That made the decision rule a
de facto constant, which is the wrong place to leave it.

It writes a per-country score side-file instead, so the rule becomes a
parameter that can be swept in seconds.

The France problem this is for
------------------------------
France never appears in ``train``, so its score floor is a heuristic
(``FR_DELTA``) rather than something fitted. Measured on the first full
run:

                        mean matches   p90   at cap 11
  France prediction          6.57        11      15.57%
  val_sample prediction     3.48         6       0.00%
  train ground truth        3.46         6       0.00%

Since ``F0.5 = 5R / (4P + T)``, predicting 6.57 candidates against a true
count near 3.46 scores about 0.50, where predicting at the prior scores
about 0.87. There are no France labels to tune against, so the defensible
target is the train prior: sweep the floor until the predicted count
distribution matches the one the ground truth actually has.

Usage
-----
  # report the predicted distribution at several floors, no files written
  ./venv_fresh/bin/python retune.py --country France --sweep

  # apply a floor and rewrite the submission
  ./venv_fresh/bin/python retune.py --country France \\
      --score-tau 0.75 --alpha 0.8 --out output/retuned
"""

import argparse
import collections
import math
import os
from pathlib import Path

MAX_PER_S1 = 11
UNSEEN_COUNTRIES = {"France"}
FR_DELTA = 0.10

HERE = Path(__file__).resolve().parent
PARTS = HERE / "output" / "parts"


def load_scores(country: str, parts: Path = PARTS):
    """Return {s1_id: [(cand_id, prob), ...]} sorted by descending prob."""
    path = parts / f"{country}.scores.tsv"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Scores are only written by runs started "
            f"after score persistence was added; re-run that country.")
    by_entity: dict = collections.defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            sid, cid, p = line.rstrip("\n").split("\t")
            by_entity[sid].append((cid, float(p)))
    for sid in by_entity:
        by_entity[sid].sort(key=lambda t: (-t[1], t[0]))
    return dict(by_entity)


def apply_rule(by_entity, score_tau: float, alpha: float,
               margin_tau: float, fr_delta: float, unseen: bool,
               max_per_s1: int = MAX_PER_S1):
    """Return {s1_id: [cand_id, ...]} under the relative-floor rule."""
    out = {}
    for sid, ranked in by_entity.items():
        if not ranked or ranked[0][1] < margin_tau:
            out[sid] = []
            continue
        best = ranked[0][1]
        floor = max(score_tau, alpha * best)
        if unseen:
            floor += fr_delta
        kept, seen = [], set()
        for cid, p in ranked:
            if p < floor:
                break
            if cid not in seen:
                seen.add(cid)
                kept.append(cid)
            if len(kept) >= max_per_s1:
                break
        out[sid] = kept
    return out


def distribution(kept_by_entity):
    counts = [len(v) for v in kept_by_entity.values()]
    n = len(counts)
    hist = collections.Counter(counts)
    cum = 0
    pct = {}
    for k in sorted(hist):
        cum += hist[k]
        for q in (0.5, 0.9, 0.99):
            if q not in pct and cum / n >= q:
                pct[q] = k
    return {
        "n": n,
        "empty_pct": 100.0 * hist[0] / n,
        "mean": sum(counts) / n,
        "p50": pct.get(0.5), "p90": pct.get(0.9), "p99": pct.get(0.99),
        "at_cap_pct": 100.0 * hist[max_per] / n if (max_per := MAX_PER_S1) else 0.0,
    }


# The target the prediction should look like, measured from
# dataset/train/train_ground_truth.tsv: 5.58% singletons, mean 3.46,
# p50 3, p90 6, p99 8, max 11. Identical for US and India.
TRAIN_PRIOR = {"empty_pct": 5.58, "mean": 3.46, "p50": 3, "p90": 6, "p99": 8}


def distance(d):
    """Rough F0.5-shaped distance from the train prior.

    F0.5 = 5R/(4P + T). Over-predicting is penalised through 4P and
    under-predicting through T, so the penalty is asymmetric and the
    weighting below is only a heuristic to rank candidates, not an
    estimate of the score.
    """
    pen = 4.0 * abs(d["mean"] - TRAIN_PRIOR["mean"]) / TRAIN_PRIOR["mean"]
    pen += 2.0 * abs(d["p90"] - TRAIN_PRIOR["p90"]) / TRAIN_PRIOR["p90"]
    pen += 0.5 * d["at_cap_pct"] / 10.0
    return pen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="France")
    ap.add_argument("--parts", type=Path, default=PARTS)
    ap.add_argument("--score-tau", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--margin-tau", type=float, default=0.01)
    ap.add_argument("--fr-delta", type=float, default=FR_DELTA)
    ap.add_argument("--sweep", action="store_true",
                    help="report the distribution across floors, write nothing")
    ap.add_argument("--out", type=Path, default=None,
                    help="write retuned matching/candidate TSVs here")
    args = ap.parse_args()

    unseen = args.country in UNSEEN_COUNTRIES
    by_entity = load_scores(args.country, args.parts)
    print(f"{args.country}: {len(by_entity):,} S1 entities with scores"
          f"{'  [unseen country: FR_DELTA applies]' if unseen else ''}")
    print(f"train prior target: mean {TRAIN_PRIOR['mean']}  p50 {TRAIN_PRIOR['p50']}"
          f"  p90 {TRAIN_PRIOR['p90']}  p99 {TRAIN_PRIOR['p99']}"
          f"  empty {TRAIN_PRIOR['empty_pct']}%\n")

    if args.sweep or args.out is None:
        print(f"{'score_tau':>9} {'alpha':>6} {'fr_delta':>9} | "
              f"{'mean':>5} {'p50':>4} {'p90':>4} {'p99':>4} "
              f"{'cap%':>6} {'empty%':>7} | penalty")
        best = None
        for st in (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97):
            for al in (0.7, 0.8, 0.9):
                kept = apply_rule(by_entity, st, al, args.margin_tau,
                                  args.fr_delta, unseen)
                d = distribution(kept)
                pen = distance(d)
                mark = ""
                if best is None or pen < best[0]:
                    best = (pen, st, al, d)
                    mark = "  <-"
                print(f"{st:9.2f} {al:6.2f} {args.fr_delta:9.2f} | "
                      f"{d['mean']:5.2f} {str(d['p50']):>4} {str(d['p90']):>4} "
                      f"{str(d['p99']):>4} {d['at_cap_pct']:6.2f} "
                      f"{d['empty_pct']:7.2f} | {pen:6.3f}{mark}")
        pen, st, al, d = best
        print(f"\nclosest to prior: score_tau={st} alpha={al} "
              f"-> mean {d['mean']:.2f} p90 {d['p90']} "
              f"cap {d['at_cap_pct']:.2f}% (penalty {pen:.3f})")
        if args.out is None:
            return 0
        args.score_tau, args.alpha = st, al

    kept = apply_rule(by_entity, args.score_tau, args.alpha, args.margin_tau,
                      args.fr_delta, unseen)
    d = distribution(kept)
    print(f"\napplied score_tau={args.score_tau} alpha={args.alpha} "
          f"fr_delta={args.fr_delta} margin={args.margin_tau}")
    print(f"  mean {d['mean']:.2f}  p50 {d['p50']}  p90 {d['p90']}  "
          f"p99 {d['p99']}  at_cap {d['at_cap_pct']:.2f}%  empty {d['empty_pct']:.2f}%")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        src = args.parts
        with open(args.out / "matching_results.tsv", "w", encoding="utf-8") as m:
            m.write("source1_entity_id\tmatched_entity_ids\n")
            for sid in sorted(kept):
                m.write(f"{sid}\t{','.join(kept[sid])}\n")
        with open(args.out / "candidate_pairs.tsv", "w", encoding="utf-8") as c:
            c.write("source1_entity_id\tcandidate_entity_ids\n")
            for sid in sorted(by_entity):
                c.write(f"{sid}\t"
                        f"{','.join(sorted(cid for cid, _ in by_entity[sid]))}\n")
        print(f"\nwrote {args.out}/matching_results.tsv and candidate_pairs.tsv")
        print("NOTE: this covers one country only. Concatenate with the other"
              " countries' part files to rebuild a full submission.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
