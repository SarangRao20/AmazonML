"""Vectorised macro-F_0.5 threshold search.

The existing grid search in ``threshold_optimizer.grid_search`` is
O(combinations x pairs) in pure Python: for every (score_tau, margin_tau)
pair it rebuilds the prediction dicts and recomputes per-entity F_0.5 from
scratch. On 1.75M pairs x 60 combinations that is over an hour.

This module does the same computation with numpy. The key observation is
that changing ``score_tau`` does not change the *ranking* of candidates
within an entity, only how many of the top-ranked ones are kept. So for
each entity we sort once by descending probability and cache:

    sorted_probs   descending probabilities
    cum_tp[k]      number of true matches among the top-k candidates
    n_true         total number of true matches for the entity

Then for any threshold, ``kept = searchsorted(sorted_probs, tau, 'right')``
gives the cut index in O(log n), and precision/recall/F follow directly.
A full 60-point grid collapses to a handful of vectorised passes.
"""

from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


class VectorizedThresholdOptimizer:
    """Grid-search (score_tau, margin_tau) against macro-averaged F_0.5.

    Numerically equivalent to the reference implementation, but runs in
    seconds instead of hours. Verified by ``verify_equivalence`` in
    ``tests/test_threshold_equivalence.py``.
    """

    def __init__(self, beta: float = 0.5) -> None:
        self.beta = beta
        b2 = beta * beta
        # F_beta = (1 + b^2) * P * R / (b^2 * P + R)
        self.b2 = b2
        self.num = 1.0 + b2
        self.den_p = b2

    # ---------------------------------------------------------------- build

    @staticmethod
    def build_entity_index(
        probabilities: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, Set[str]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flatten per-entity candidates into CSR-style sorted arrays.

        Args:
            probabilities: S1 id -> list of (candidate_id, probability).
            ground_truth:  S1 id -> set of true matching candidate ids.

        Returns:
            indptr      (n_entities + 1,)  offsets into the flat arrays
            sorted_prob (n_pairs,)         probabilities, descending per entity
            cum_tp      (n_pairs,)         true positives among the top-k
            n_true      (n_entities,)      true match count per entity
            max_prob    (n_entities,)      best probability per entity
        """
        entity_ids = sorted(ground_truth.keys())

        flat_prob: List[np.ndarray] = []
        cum_tp_list: List[np.ndarray] = []
        n_true = np.zeros(len(entity_ids), dtype=np.float64)
        max_prob = np.zeros(len(entity_ids), dtype=np.float64)
        indptr = np.zeros(len(entity_ids) + 1, dtype=np.int64)

        for i, s1_id in enumerate(entity_ids):
            cands = probabilities.get(s1_id, [])
            truth = ground_truth[s1_id]

            if not cands:
                indptr[i + 1] = indptr[i]
                continue

            ids = np.fromiter((c for c, _ in cands), dtype=object, count=len(cands))
            probs = np.fromiter((p for _, p in cands), dtype=np.float64,
                                count=len(cands))
            labels = np.fromiter((1.0 if cid in truth else 0.0
                                  for cid, _ in cands),
                                 dtype=np.float64, count=len(cands))

            # Descending by probability; ties broken by candidate id so the
            # ordering is deterministic and matches the reference code.
            order = np.lexsort((ids, -probs))
            probs = probs[order]
            labels = labels[order]

            flat_prob.append(probs)
            # cum_tp[k] = true positives among the first k+1 candidates
            cum_tp_list.append(np.cumsum(labels))

            n_true[i] = float(len(truth))
            max_prob[i] = probs[0] if probs.size else 0.0
            indptr[i + 1] = indptr[i] + probs.size

        if flat_prob:
            sorted_prob = np.concatenate(flat_prob)
            cum_tp = np.concatenate(cum_tp_list)
        else:
            sorted_prob = np.zeros(0, dtype=np.float64)
            cum_tp = np.zeros(0, dtype=np.float64)

        return indptr, sorted_prob, cum_tp, n_true, max_prob

    # --------------------------------------------------------------- scoring

    def macro_f_beta_for(
        self,
        indptr: np.ndarray,
        sorted_prob: np.ndarray,
        cum_tp: np.ndarray,
        n_true: np.ndarray,
        max_prob: np.ndarray,
        score_tau: float,
        margin_tau: float,
    ) -> float:
        """Macro-average F_beta for one threshold pair.

        Decision rule, matching ``apply_thresholds``:
          1. if the entity's best probability < margin_tau -> predict empty
             (scores 1.0 if the entity is a true singleton, else 0.0)
          2. otherwise keep every candidate with probability >= score_tau
        """
        starts = indptr[:-1]
        ends = indptr[1:]
        sizes = ends - starts

        # Number of candidates kept per entity. Within an entity the block is
        # sorted descending, so {p >= tau} is a prefix and the kept count is
        # just the number of True values in the block. A global cumsum
        # differenced at the block boundaries gives every count in one pass.
        kept = np.zeros(len(starts), dtype=np.int64)
        if sorted_prob.size:
            flags = (sorted_prob >= score_tau).astype(np.int64)
            running = np.concatenate(([0], np.cumsum(flags)))
            kept = running[ends] - running[starts]

        # True positives captured among the kept prefix. `cum_tp` is a
        # per-block cumulative sum of the labels, so the count inside the
        # first `kept` entries of block [s, e) is cum_tp[s + kept - 1].
        tp = np.zeros(len(starts), dtype=np.float64)
        has_kept = kept > 0
        if has_kept.any():
            idx = starts[has_kept] + kept[has_kept] - 1
            tp[has_kept] = cum_tp[idx]

        # Singletons: empty truth. Predicted empty -> 1.0, else -> 0.0.
        is_singleton = n_true == 0
        predicts_empty = (~has_kept) | (max_prob < margin_tau)

        f = np.zeros(len(starts), dtype=np.float64)
        # Correctly predicted singleton.
        f[is_singleton & predicts_empty] = 1.0
        # Singleton that we wrongly gave a match to.
        f[is_singleton & ~predicts_empty] = 0.0

        # Non-singleton entities.
        non_single = ~is_singleton
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(kept > 0, tp / np.maximum(kept, 1), 0.0)
            recall = tp / np.maximum(n_true, 1.0)
            fb = (self.num * precision * recall) / (
                self.den_p * precision + recall
            )
        fb = np.nan_to_num(fb, nan=0.0, posinf=0.0, neginf=0.0)

        active = non_single & has_kept & (max_prob >= margin_tau)
        f[active] = fb[active]
        # Non-singleton where we predicted empty -> 0.0 (already zero).

        return float(f.mean())

    # ------------------------------------------------------------------ grid

    def grid_search(
        self,
        probabilities: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, Set[str]],
        score_range: Tuple[float, float] = (0.30, 0.95),
        score_step: float = 0.01,
        margin_range: Tuple[float, float] = (0.05, 0.50),
        margin_step: float = 0.01,
        top_k: int = 10,
    ) -> Tuple[float, float, float, List[dict]]:
        """Search the (score_tau, margin_tau) grid for maximum macro F_beta."""
        index = self.build_entity_index(probabilities, ground_truth)

        score_grid = np.round(
            np.arange(score_range[0], score_range[1] + score_step / 2, score_step), 4
        )
        margin_grid = np.round(
            np.arange(margin_range[0], margin_range[1] + margin_step / 2, margin_step), 4
        )

        results: List[dict] = []
        best = (-1.0, score_grid[0], margin_grid[0])

        for score_tau in score_grid:
            for margin_tau in margin_grid:
                value = self.macro_f_beta_for(*index, float(score_tau),
                                             float(margin_tau))
                results.append({
                    "score_threshold": float(score_tau),
                    "margin_threshold": float(margin_tau),
                    "macro_f_beta": value,
                })
                if value > best[0]:
                    best = (value, float(score_tau), float(margin_tau))

        results.sort(key=lambda r: r["macro_f_beta"], reverse=True)
        return best[1], best[2], best[0], results[:top_k]

    # ----------------------------------------------------------------- apply

    def apply_thresholds(
        self,
        probabilities: Dict[str, List[Tuple[str, float]]],
        score_tau: float,
        margin_tau: float,
    ) -> Dict[str, Set[str]]:
        """Materialise predictions for a threshold pair (used at inference)."""
        out: Dict[str, Set[str]] = {}
        for s1_id, cands in probabilities.items():
            if not cands:
                out[s1_id] = set()
                continue
            if max(p for _, p in cands) < margin_tau:
                out[s1_id] = set()
                continue
            out[s1_id] = {c for c, p in cands if p >= score_tau}
        return out
