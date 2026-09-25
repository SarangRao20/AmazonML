"""Prove the vectorised threshold search matches the reference implementation.

A fast-but-different threshold search is worse than a slow correct one, so
this compares the two on randomised data across many threshold pairs,
including the edge cases: singletons, entities with no candidates, perfect
matches, and zero-match entities.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.business_entity_resolution.src.threshold_optimizer import ThresholdOptimizer
from code.business_entity_resolution.src.threshold_optimizer_fast import (
    VectorizedThresholdOptimizer,
)


def make_case(seed: int, n_entities: int = 400):
    rng = random.Random(seed)
    probabilities = {}
    ground_truth = {}
    for i in range(n_entities):
        s1 = f"S1-{i:05d}"
        n_cand = rng.choice([0, 1, 1, 2, 3, 5, 8, 12, 30])
        cands = []
        for j in range(n_cand):
            p = rng.choice([
                rng.random(),          # uniform
                min(1.0, rng.random() * 0.3),
                1.0 - rng.random() * 0.05,  # clustered near 1
            ])
            cands.append((f"S2-{i:05d}-{j:03d}", round(p, 6)))
        # Ground truth: sometimes empty (singleton), sometimes a subset.
        mode = rng.random()
        if mode < 0.15 or not cands:
            truth = set()
        elif mode < 0.45:
            truth = {cands[0][0]}  # exactly one true match
        else:
            k = rng.randint(1, len(cands))
            truth = {c for c, _ in cands[:k]}
        probabilities[s1] = cands
        ground_truth[s1] = truth
    return probabilities, ground_truth


def main() -> int:
    slow = ThresholdOptimizer(verbose=False)
    fast = VectorizedThresholdOptimizer(beta=0.5)

    grid = [(s, m)
            for s in (0.30, 0.45, 0.55, 0.65, 0.75, 0.83, 0.90, 0.95)
            for m in (0.05, 0.20, 0.35, 0.50)]

    worst = 0.0
    checked = 0
    for seed in range(12):
        probabilities, ground_truth = make_case(seed)
        index = fast.build_entity_index(probabilities, ground_truth)
        for score_tau, margin_tau in grid:
            ref = slow.compute_macro_f_beta(
                slow.apply_thresholds(probabilities, score_tau, margin_tau),
                ground_truth, beta=0.5,
            )
            got = fast.macro_f_beta_for(*index, score_tau, margin_tau)
            worst = max(worst, abs(ref - got))
            checked += 1
            if abs(ref - got) > 1e-9:
                print(f"MISMATCH seed={seed} tau=({score_tau},{margin_tau}) "
                      f"ref={ref:.12f} fast={got:.12f}")
                return 1

    print(f"compared {checked} (case, threshold) pairs across 12 random cases")
    print(f"max absolute difference: {worst:.3e}")

    # Also confirm the applied predictions themselves match, not just the score.
    probabilities, ground_truth = make_case(99)
    for score_tau, margin_tau in [(0.65, 0.20), (0.90, 0.35), (0.45, 0.05)]:
        a = slow.apply_thresholds(probabilities, score_tau, margin_tau)
        b = fast.apply_thresholds(probabilities, score_tau, margin_tau)
        if a != b:
            print(f"PREDICTION MISMATCH at ({score_tau},{margin_tau})")
            return 1
    print("applied prediction sets identical")

    # Timing on a realistically sized case.
    probabilities, ground_truth = make_case(7, n_entities=40_000)
    import time
    index = fast.build_entity_index(probabilities, ground_truth)
    t0 = time.time()
    for score_tau in [i / 100 for i in range(30, 96, 2)]:
        for margin_tau in [i / 100 for i in range(5, 51, 5)]:
            fast.macro_f_beta_for(*index, score_tau, margin_tau)
    fast_dt = time.time() - t0

    t0 = time.time()
    for score_tau in [i / 100 for i in range(30, 96, 2)]:
        for margin_tau in [i / 100 for i in range(5, 51, 5)]:
            slow.compute_macro_f_beta(
                slow.apply_thresholds(probabilities, score_tau, margin_tau),
                ground_truth, beta=0.5)
    slow_dt = time.time() - t0

    n_pairs = sum(len(v) for v in probabilities.values())
    print(f"\n40,000 entities / {n_pairs:,} pairs / 297 grid points:")
    print(f"  reference: {slow_dt:8.2f}s")
    print(f"  vectorised:{fast_dt:8.2f}s   ({slow_dt / max(fast_dt, 1e-9):.0f}x faster)")
    print("\nEQUIVALENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
