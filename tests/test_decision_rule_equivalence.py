"""Prove the vectorised decision rules match naive reference implementations.

Each rule gets a straightforward, obviously-correct loop version here, and
the two are compared on randomised data that deliberately includes the
awkward cases: true singletons, entities with no candidates, entities whose
true matches sit below low-probability candidates, ties in probability, and
zero-probability candidates.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from code.business_entity_resolution.src.decision_rule import (
    DecisionRuleOptimizer, EntityIndex, EPS,
)


# --------------------------------------------------------------------------
# Naive reference implementations
# --------------------------------------------------------------------------

def ref_entity_f(y_true, y_pred, beta=0.5):
    if not y_true and not y_pred:
        return 1.0
    if not y_true or not y_pred:
        return 0.0
    tp = len(y_true & y_pred)
    fp = len(y_pred - y_true)
    fn = len(y_true - y_pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return ((1 + b2) * p * r) / (b2 * p + r)


def ref_macro(probabilities, ground_truth, pred_fn, beta=0.5):
    scores = []
    for s1_id in ground_truth:
        y_true = ground_truth[s1_id]
        pred = pred_fn(s1_id, probabilities.get(s1_id, []))
        scores.append(ref_entity_f(y_true, pred, beta))
    return float(np.mean(scores)) if scores else 0.0


def ref_pred_threshold(s1_id, cands, score_tau, margin_tau):
    if not cands:
        return set()
    if max(p for _, p in cands) < margin_tau:
        return set()
    return {c for c, p in cands if p >= score_tau}


def ref_pred_relative(s1_id, cands, score_tau, alpha, margin_tau):
    if not cands:
        return set()
    max_p = max(p for _, p in cands)
    if max_p < margin_tau:
        return set()
    floor = max(score_tau, alpha * max_p)
    return {c for c, p in cands if p >= floor}


def ref_pred_expected(s1_id, cands, truth, shift, score_tau, beta=0.5):
    """Reference expected-F_beta top-k, written as plainly as possible."""
    if not cands:
        return set()
    b2 = beta * beta
    num = 1.0 + b2
    # Same ordering as EntityIndex: descending probability, ties by id.
    ordered = sorted(cands, key=lambda t: (-t[1], t[0]))
    if shift:
        q = [1.0 / (1.0 + np.exp(-(np.log(min(max(p, EPS), 1 - EPS) /
                                           (1 - min(max(p, EPS), 1 - EPS)))
                                       + shift))) for _, p in ordered]
    else:
        q = [min(max(p, EPS), 1 - EPS) for _, p in ordered]

    e_none = 1.0
    for qi in q:
        e_none *= (1.0 - qi)

    s_n = sum(q)
    best_k, best_f = 0, e_none
    running = 0.0
    for k, qi in enumerate(q, start=1):
        running += qi
        p = running / k
        r = running / s_n if s_n > 0 else 0.0
        f = 0.0 if (p + r) == 0 else (num * p * r) / (b2 * p + r)
        if f > best_f:
            best_f, best_k = f, k

    if score_tau > 0:
        best_k = min(best_k, sum(1 for _, p in ordered if p >= score_tau))
    return {c for c, _ in ordered[:best_k]}


# --------------------------------------------------------------------------
# Randomised comparison
# --------------------------------------------------------------------------

def make_case(seed, n_entities=300):
    rng = random.Random(seed)
    probabilities, ground_truth = {}, {}
    for i in range(n_entities):
        s1 = f"S1-{i:05d}"
        n_cand = rng.choice([0, 1, 1, 2, 3, 4, 6, 10, 20])
        cands = []
        for j in range(n_cand):
            r = rng.random()
            if r < 0.10:
                p = 0.0                      # zero-probability candidate
            elif r < 0.20:
                p = 1.0                      # saturated
            elif r < 0.30:
                p = 0.5                      # tie-inducing value
            elif r < 0.6:
                p = round(rng.random() * 0.3, 6)
            else:
                p = round(1.0 - rng.random() * 0.05, 6)
            cands.append((f"S2-{i:05d}-{j:03d}", p))
        # duplicates in probability to exercise tie-breaking
        if cands and rng.random() < 0.25 and len(cands) > 2:
            cands[2] = (cands[2][0], cands[1][1])

        mode = rng.random()
        if mode < 0.20 or not cands:
            truth = set()
        elif mode < 0.45:
            # true match deliberately placed at a LOW probability
            truth = {cands[-1][0]}
        elif mode < 0.70:
            truth = {cands[0][0]}
        else:
            k = rng.randint(1, len(cands))
            truth = {c for c, _ in cands[:k]}
        probabilities[s1] = cands
        ground_truth[s1] = truth
    return probabilities, ground_truth


def main() -> int:
    opt = DecisionRuleOptimizer(beta=0.5)
    failures = 0
    checked = 0

    score_grid = [0.30, 0.50, 0.65, 0.72, 0.85, 0.95]
    margin_grid = [0.01, 0.10, 0.25, 0.40]
    alpha_grid = [0.2, 0.5, 0.7, 0.9]

    for seed in range(15):
        probabilities, ground_truth = make_case(seed)
        index = EntityIndex(probabilities, ground_truth)

        for score_tau in score_grid:
            for margin_tau in margin_grid:
                got = opt.macro_for(index, "threshold",
                                    score_tau=score_tau, margin_tau=margin_tau)
                exp = ref_macro(
                    probabilities, ground_truth,
                    lambda s, c, st=score_tau, mt=margin_tau:
                        ref_pred_threshold(s, c, st, mt))
                checked += 1
                if abs(got - exp) > 1e-9:
                    print(f"THRESHOLD MISMATCH seed={seed} ({score_tau},{margin_tau}) "
                          f"fast={got:.12f} ref={exp:.12f}")
                    failures += 1

                for alpha in alpha_grid:
                    got_r = opt.macro_for(index, "relative",
                                          score_tau=score_tau, alpha=alpha,
                                          margin_tau=margin_tau)
                    exp_r = ref_macro(
                        probabilities, ground_truth,
                        lambda s, c, st=score_tau, a=alpha, mt=margin_tau:
                            ref_pred_relative(s, c, st, a, mt))
                    checked += 1
                    if abs(got_r - exp_r) > 1e-9:
                        print(f"RELATIVE MISMATCH seed={seed} "
                              f"({score_tau},{alpha},{margin_tau}) "
                              f"fast={got_r:.12f} ref={exp_r:.12f}")
                        failures += 1

        for shift in (-1.5, -0.5, 0.0, 0.5, 1.5):
            for score_tau in (0.0, 0.2, 0.4):
                got_e = opt.macro_for(index, "expected", shift=shift,
                                      score_tau=score_tau)
                exp_e = ref_macro(
                    probabilities, ground_truth,
                    lambda s, c, sh=shift, st=score_tau: ref_pred_expected(
                        s, c, ground_truth.get(s, set()), sh, st))
                checked += 1
                if abs(got_e - exp_e) > 1e-9:
                    print(f"EXPECTED MISMATCH seed={seed} (shift={shift}, "
                          f"st={score_tau}) fast={got_e:.12f} ref={exp_e:.12f}")
                    failures += 1

    print(f"compared {checked} (rule, case, params) combinations")
    print(f"mismatches: {failures}")

    # Prediction sets must match too, not just the scalar.
    probabilities, ground_truth = make_case(123)
    index = EntityIndex(probabilities, ground_truth)
    pred_mismatch = 0
    for score_tau, alpha, margin_tau in [(0.5, 0.5, 0.1), (0.72, 0.7, 0.25)]:
        a = opt.apply(index, "threshold", score_tau=score_tau, margin_tau=margin_tau)
        for s1_id, cands in probabilities.items():
            if a[s1_id] != ref_pred_threshold(s1_id, cands, score_tau, margin_tau):
                pred_mismatch += 1
        b = opt.apply(index, "relative", score_tau=score_tau, alpha=alpha,
                      margin_tau=margin_tau)
        for s1_id, cands in probabilities.items():
            if b[s1_id] != ref_pred_relative(s1_id, cands, score_tau, alpha, margin_tau):
                pred_mismatch += 1
    for shift in (-0.5, 0.0, 0.5):
        c = opt.apply(index, "expected", shift=shift, score_tau=0.0)
        for s1_id, cands in probabilities.items():
            exp = ref_pred_expected(s1_id, cands, ground_truth.get(s1_id, set()),
                                    shift, 0.0)
            if c[s1_id] != exp:
                pred_mismatch += 1
    print(f"prediction-set mismatches: {pred_mismatch}")

    if failures or pred_mismatch:
        print("\nFAILED")
        return 1
    print("\nALL EQUIVALENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
