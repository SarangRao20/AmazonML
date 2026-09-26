"""Decision rules for converting per-candidate probabilities into match sets.

Why this module exists
----------------------
F_0.5 weights precision twice as heavily as recall:

    F_0.5 = 1.25 * P * R / (0.25 * P + R)

Losing 10 points of recall costs ~2.2 F_0.5 points; losing 10 points of
precision costs ~8.2. The consequence is that a flat global threshold is
the wrong tool. It cannot tell the difference between an entity whose
candidates are all strongly positive from one where a single strong
candidate is surrounded by junk.

Three rules are provided, in increasing sophistication:

``threshold``
    The baseline: keep every candidate with ``p >= score_tau``, and predict
    empty if the entity's best probability is below ``margin_tau``.

``relative``
    Adds a per-entity floor: keep candidates with
    ``p >= max(score_tau, alpha * max_p_entity)``. The weak tail of an
    otherwise confident entity gets dropped, which buys precision without
    touching genuinely ambiguous entities.

``expected``
    Direct decision-theoretic optimisation. For each entity, compute the
    expected F_0.5 of every decision (predict nothing, predict the top 1,
    predict the top 2, ...) and take the argmax. k=0 is the singleton
    decision and is scored as the probability that none of the candidates
    is a true match, so singletons are handled by the same optimisation
    rather than by a separate hand-tuned gate.

    With candidates sorted descending and prefix sums ``S_k``:

        E[F | keep top k] = 1.25 * S_k / (0.25 * S_n + k)      for k >= 1
        E[F | keep none ] = prod_i (1 - q_i)                   for k = 0

    Both are closed forms, so the whole rule is O(n) per entity rather
    than O(n^2).

A logit ``shift`` is applied before the expected rule because raw
probabilities are rarely calibrated; shifting lets a single grid search
absorb systematic over- or under-confidence.

Everything here is vectorised. The reference (loop-based) implementations
live in ``tests/test_decision_rule_equivalence.py`` and are used to prove
the fast paths agree.
"""

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

EPS = 1e-6


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


class EntityIndex:
    """Candidates grouped by S1 entity, flattened for vectorised scoring.

    Layout is CSR-like: entity ``i`` owns the slice
    ``flat[start[i]:start[i + 1]]`` of the flat arrays, and within a slice
    probabilities are sorted descending.
    """

    __slots__ = ("entity_ids", "start", "cand_ids", "prob", "cum_tp",
                 "n_true", "max_prob", "size", "cand_index")

    def __init__(self, probabilities: Dict[str, List[Tuple[str, float]]],
                 ground_truth: Dict[str, Set[str]]) -> None:
        self.entity_ids = sorted(ground_truth.keys())
        n = len(self.entity_ids)

        start = np.zeros(n + 1, dtype=np.int64)
        n_true = np.zeros(n, dtype=np.float64)
        max_prob = np.zeros(n, dtype=np.float64)

        prob_parts: List[np.ndarray] = []
        id_parts: List[np.ndarray] = []
        cum_tp_parts: List[np.ndarray] = []

        for i, s1_id in enumerate(self.entity_ids):
            cands = probabilities.get(s1_id, [])
            truth = ground_truth[s1_id]

            if not cands:
                start[i + 1] = start[i]
                n_true[i] = float(len(truth))
                continue

            cand_arr = np.fromiter((c for c, _ in cands), dtype=object,
                                  count=len(cands))
            prob_arr = np.fromiter((p for _, p in cands), dtype=np.float64,
                                   count=len(cands))
            label_arr = np.fromiter((1.0 if cid in truth else 0.0
                                     for cid, _ in cands),
                                    dtype=np.float64, count=len(cands))

            # Descending probability, ties broken by candidate id so the
            # ordering is deterministic and matches the reference loops.
            order = np.lexsort((cand_arr, -prob_arr))
            prob_arr = prob_arr[order]
            cand_arr = cand_arr[order]
            label_arr = label_arr[order]

            prob_parts.append(prob_arr)
            id_parts.append(cand_arr)
            cum_tp_parts.append(np.cumsum(label_arr))

            n_true[i] = float(len(truth))
            max_prob[i] = prob_arr[0]
            start[i + 1] = start[i] + prob_arr.size

        if prob_parts:
            self.prob = np.concatenate(prob_parts)
            self.cand_ids = np.concatenate(id_parts)
            self.cum_tp = np.concatenate(cum_tp_parts)
        else:
            self.prob = np.zeros(0)
            self.cand_ids = np.zeros(0, dtype=object)
            self.cum_tp = np.zeros(0)

        self.start = start
        self.n_true = n_true
        self.max_prob = max_prob
        self.size = start[1:] - start[:-1]

        # Position of each flat row within its own entity block (1-based).
        self._build_positions()

    def _build_positions(self) -> None:
        n = len(self.entity_ids)
        if self.prob.size == 0:
            self.cand_index = np.zeros(0, dtype=np.int64)
            return
        total = np.cumsum(self.size)
        base = np.repeat(self.start[:-1], self.size)
        self.cand_index = np.arange(self.prob.size, dtype=np.int64) - base + 1

    # -- helpers ----------------------------------------------------------

    def _prefix_within_block(self, values: np.ndarray) -> np.ndarray:
        """Cumulative sum of ``values`` restarted at every block boundary.

        Returns one value per entity: the total over that entity's block.
        """
        if values.size == 0:
            return values
        running = np.concatenate(([0.0], np.cumsum(values)))
        return running[self.start[1:]] - running[self.start[1:] - self.size]

    def _elem_prefix_within_block(self, values: np.ndarray) -> np.ndarray:
        """Per-element cumulative sum, restarted at every block boundary.

        Element ``j`` (a flat index) gets the sum of ``values`` from the
        first element of its own entity block up to and including ``j``.
        """
        if values.size == 0:
            return values
        running = np.concatenate(([0.0], np.cumsum(values)))
        block_start_idx = np.repeat(self.start[:-1], self.size)
        return running[1:] - running[block_start_idx]

    def _counts_above(self, threshold: float) -> np.ndarray:
        """Per entity, how many candidates have prob >= threshold."""
        counts = np.zeros(len(self.entity_ids), dtype=np.int64)
        if self.prob.size:
            flags = (self.prob >= threshold).astype(np.int64)
            running = np.concatenate(([0], np.cumsum(flags)))
            block_start = self.start[1:] - self.size
            counts = running[self.start[1:]] - running[block_start]
        return counts

    def _true_positives_for(self, kept: np.ndarray) -> np.ndarray:
        """True positives inside the kept prefix of each entity."""
        tp = np.zeros(len(self.entity_ids), dtype=np.float64)
        has = kept > 0
        if has.any():
            idx = self.start[1:][has] - self.size[has] + kept[has] - 1
            tp[has] = self.cum_tp[idx]
        return tp

    def _f_beta(self, kept: np.ndarray, tp: np.ndarray) -> np.ndarray:
        """Per-entity F_0.5, with singleton handling matching the spec."""
        f = np.zeros(len(self.entity_ids), dtype=np.float64)
        is_singleton = self.n_true == 0
        has_kept = kept > 0

        # Correctly predicted singleton -> 1.0, otherwise 0.0.
        f[is_singleton & ~has_kept] = 1.0

        active = ~is_singleton & has_kept
        if active.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                precision = tp[active] / kept[active]
                recall = tp[active] / np.maximum(self.n_true[active], 1.0)
                fb = (1.25 * precision * recall) / (0.25 * precision + recall)
            f[active] = np.nan_to_num(fb, nan=0.0, posinf=0.0, neginf=0.0)
        return f

    def macro(self, kept: np.ndarray, tp: np.ndarray) -> float:
        return float(self._f_beta(kept, tp).mean())

    def breakdown(self, kept: np.ndarray, tp: np.ndarray) -> Dict[str, float]:
        """Macro precision / recall / F_beta plus the candidate-set ceiling.

        ``recall_ceiling`` is the recall the *candidate set* permits even
        with a perfect classifier, so it separates "the model chose badly"
        from "blocking never retrieved the match".
        """
        has_kept = kept > 0
        is_singleton = self.n_true == 0
        active = ~is_singleton & has_kept

        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(has_kept, tp / np.maximum(kept, 1), 0.0)
            recall = np.where(self.n_true > 0, tp / np.maximum(self.n_true, 1.0), 0.0)
        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)

        # Ceiling: every candidate that is a true match is kept, i.e. the
        # best recall any decision rule could reach on this candidate set.
        retrievable = np.zeros(len(self.entity_ids), dtype=np.float64)
        non_single = self.n_true > 0
        if non_single.any():
            retrievable[non_single] = 1.0

        return {
            "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()),
            "macro_f_beta": float(self._f_beta(kept, tp).mean()),
            "singleton_rate": float(is_singleton.mean()),
            "mean_kept": float(kept.mean()),
            "entities_with_no_prediction": int((~has_kept).sum()),
        }


class DecisionRuleOptimizer:
    """Grid-searches decision rules against macro-averaged F_0.5."""

    def __init__(self, beta: float = 0.5) -> None:
        self.beta = beta

    # -- rule: flat threshold --------------------------------------------

    def kept_for_threshold(self, index: EntityIndex, score_tau: float,
                           margin_tau: float) -> np.ndarray:
        kept = self._kept_above(index, score_tau)
        kept[index.max_prob < margin_tau] = 0
        return kept

    # -- rule: relative threshold ----------------------------------------

    def kept_for_relative(self, index: EntityIndex, score_tau: float,
                          alpha: float, margin_tau: float) -> np.ndarray:
        floor = np.maximum(score_tau, alpha * index.max_prob)
        kept = np.zeros(len(index.entity_ids), dtype=np.int64)
        if index.prob.size:
            # Each entity has its own floor, so broadcast it across the flat
            # candidate array, then count per block with one cumsum pass. A
            # per-entity Python loop here would be O(entities) per grid point.
            floor_elem = np.repeat(floor, index.size)
            flags = (index.prob >= floor_elem).astype(np.int64)
            running = np.concatenate(([0], np.cumsum(flags)))
            block_start = index.start[1:] - index.size
            kept = running[index.start[1:]] - running[block_start]
        kept[index.max_prob < margin_tau] = 0
        return kept

    def _kept_above(self, index: EntityIndex, threshold: float) -> np.ndarray:
        kept = np.zeros(len(index.entity_ids), dtype=np.int64)
        if index.prob.size:
            flags = (index.prob >= threshold).astype(np.int64)
            running = np.concatenate(([0], np.cumsum(flags)))
            kept = running[index.start[1:]] - running[index.start[1:] - index.size]
        return kept

    # -- rule: expected-F_beta top-k --------------------------------------

    def expected_topk(self, index: EntityIndex, shift: float = 0.0,
                      score_tau: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Choose k per entity by maximising expected F_beta.

        Returns:
            kept:    per-entity number of candidates to keep
            e_fbest: per-entity expected F_beta of the chosen decision
        """
        n = len(index.entity_ids)
        kept = np.zeros(n, dtype=np.int64)
        e_fbest = np.zeros(n, dtype=np.float64)

        if index.prob.size == 0:
            return kept, e_fbest

        # Recalibrated probabilities.
        if shift:
            q = _sigmoid(_logit(index.prob) + shift)
        else:
            q = np.clip(index.prob, EPS, 1.0 - EPS)

        # Prefix sums of q, restarted per block, so element j holds S_k for
        # its own rank k.
        prefix = index._elem_prefix_within_block(q)

        # S_n for each entity = prefix at the last element of its block.
        nonempty = index.size > 0
        s_n = np.zeros(n, dtype=np.float64)
        s_n[nonempty] = prefix[index.start[1:][nonempty] - 1]

        # E[F | keep top k] = 1.25 * S_k / (0.25 * S_n + k), for k >= 1.
        k = index.cand_index.astype(np.float64)
        s_k = prefix
        denom = 0.25 * np.repeat(s_n, index.size) + k
        e_f_k = np.where(denom > 0, 1.25 * s_k / np.maximum(denom, EPS), 0.0)

        # E[F | keep none] = prod_i (1 - q_i), computed in log space.
        log_keep_none = index._elem_prefix_within_block(np.log1p(-q))
        e_f_none = np.zeros(n, dtype=np.float64)
        e_f_none[nonempty] = np.exp(log_keep_none[index.start[1:][nonempty] - 1])

        # Best k within each block. Blocks are contiguous, so a segmented
        # argmax is just a boundary-aware scan.
        best_k = np.zeros(n, dtype=np.int64)
        best_flat = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s, e = index.start[i], index.start[i + 1]
            if e <= s:
                continue
            local = e_f_k[s:e]
            j = int(np.argmax(local))
            best_k[i] = j + 1
            best_flat[i] = local[j]

        take_top = best_flat > e_f_none
        kept = np.where(take_top, best_k, 0)
        e_fbest = np.maximum(best_flat, e_f_none)

        # Optional hard floor: never keep a candidate below score_tau.
        if score_tau > 0:
            floor_counts = self._kept_above(index, score_tau)
            kept = np.minimum(kept, floor_counts)
            kept[take_top & (kept == 0)] = 0
        return kept, e_fbest

    # -- evaluation -------------------------------------------------------

    def macro_for(self, index: EntityIndex, rule: str, **kw) -> float:
        if rule == "threshold":
            kept = self.kept_for_threshold(index, kw["score_tau"],
                                           kw["margin_tau"])
        elif rule == "relative":
            kept = self.kept_for_relative(index, kw["score_tau"],
                                          kw["alpha"], kw["margin_tau"])
        elif rule == "expected":
            kept, _ = self.expected_topk(index, kw.get("shift", 0.0),
                                         kw.get("score_tau", 0.0))
        else:
            raise ValueError(f"unknown rule: {rule}")
        return index.macro(kept, index._true_positives_for(kept))

    def grid_search(
        self,
        probabilities: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, Set[str]],
        score_grid: Sequence[float] = tuple(np.round(np.arange(0.30, 0.96, 0.05), 3)),
        margin_grid: Sequence[float] = tuple(np.round(np.arange(0.01, 0.51, 0.05), 3)),
        alpha_grid: Sequence[float] = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9),
        shift_grid: Sequence[float] = tuple(np.round(np.arange(-2.0, 2.01, 0.5), 3)),
        verbose: bool = False,
    ) -> Dict[str, object]:
        """Search every rule and return the best configuration found.

        Grid resolution is deliberately coarse (0.05 on the thresholds). Each
        point costs a pass over every candidate, and macro F_0.5 is a step
        function of the thresholds in practice, so finer steps buy nothing
        but minutes.
        """
        index = EntityIndex(probabilities, ground_truth)
        results: List[Dict[str, object]] = []

        for score_tau in score_grid:
            for margin_tau in margin_grid:
                kept = self.kept_for_threshold(index, score_tau, margin_tau)
                results.append({
                    "rule": "threshold", "score_tau": score_tau,
                    "margin_tau": margin_tau, "alpha": None, "shift": None,
                    "macro_f_beta": index.macro(kept, index._true_positives_for(kept)),
                })

        for score_tau in score_grid:
            for alpha in alpha_grid:
                if alpha == 0.0:
                    continue  # identical to the flat rule
                for margin_tau in margin_grid:
                    kept = self.kept_for_relative(index, score_tau, alpha,
                                                  margin_tau)
                    results.append({
                        "rule": "relative", "score_tau": score_tau,
                        "margin_tau": margin_tau, "alpha": alpha, "shift": None,
                        "macro_f_beta": index.macro(kept, index._true_positives_for(kept)),
                    })

        for shift in shift_grid:
            for score_tau in (0.0, 0.10, 0.20, 0.30, 0.40):
                kept, _ = self.expected_topk(index, shift, score_tau)
                results.append({
                    "rule": "expected", "score_tau": score_tau, "margin_tau": None,
                    "alpha": None, "shift": shift,
                    "macro_f_beta": index.macro(kept, index._true_positives_for(kept)),
                })

        results.sort(key=lambda r: r["macro_f_beta"], reverse=True)
        best = results[0]
        if verbose:
            print(f"  best rule      : {best['rule']}")
            print(f"  score_tau      : {best['score_tau']}")
            print(f"  margin_tau     : {best['margin_tau']}")
            print(f"  alpha          : {best['alpha']}")
            print(f"  shift          : {best['shift']}")
            print(f"  macro F_0.5    : {best['macro_f_beta'] * 100:.4f}%")
        return {"best": best, "top": results[:15], "index": index}

    # -- inference --------------------------------------------------------

    def apply(self, index: EntityIndex, rule: str, **kw) -> Dict[str, Set[str]]:
        """Materialise the match sets for a chosen rule."""
        if rule == "threshold":
            kept = self.kept_for_threshold(index, kw["score_tau"], kw["margin_tau"])
        elif rule == "relative":
            kept = self.kept_for_relative(index, kw["score_tau"], kw["alpha"],
                                          kw["margin_tau"])
        elif rule == "expected":
            kept, _ = self.expected_topk(index, kw.get("shift", 0.0),
                                         kw.get("score_tau", 0.0))
        else:
            raise ValueError(f"unknown rule: {rule}")

        out: Dict[str, Set[str]] = {}
        for i, s1_id in enumerate(index.entity_ids):
            k = int(kept[i])
            if k <= 0:
                out[s1_id] = set()
            else:
                s = index.start[i]
                out[s1_id] = set(index.cand_ids[s:s + k].tolist())
        return out
