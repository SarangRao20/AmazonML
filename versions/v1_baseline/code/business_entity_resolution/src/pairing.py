"""Pairing OOF predictions back to their candidate pairs, and grouping by entity.

This exists as a single shared function because getting it wrong is silent
and catastrophic. Two orderings are in play and neither is obvious:

* ``features_df`` is in the order rows were produced, i.e. grouped by S1
  entity as the blocker emitted them.
* ``oof_predictions`` is built by ``pd.concat`` over cross-validation
  folds, so it is in **fold** order.

Joining them positionally pairs every candidate with another entity's
probability. The only reliable link is the ``row_index`` column that
``TriEnsembleModel.train_groupkfold`` stamps onto each fold, which holds
the original feature-row index. Realign on that, then group.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def build_probabilities_by_s1(
    oof: pd.DataFrame,
    s1_ids: np.ndarray,
    s2_s3_ids: np.ndarray,
) -> Dict[str, List[Tuple[str, float]]]:
    """Group blended OOF probabilities by S1 entity, ranked descending.

    Args:
        oof: OOF frame carrying a ``row_index`` column and ``blend_prob``.
        s1_ids: S1 entity id per feature row, aligned to the feature index.
        s2_s3_ids: candidate id per feature row, aligned to the feature index.

    Returns:
        S1 id -> list of (candidate_id, probability) sorted by descending
        probability, ties broken by candidate id for determinism.
    """
    if "row_index" not in oof.columns:
        raise ValueError(
            "oof_predictions has no 'row_index' column, so probabilities "
            "cannot be joined to their candidate pairs. OOF rows are in fold "
            "order and the feature matrix is not, so a positional join would "
            "silently mis-pair them."
        )

    # row_index is the original feature-row index. Cast defensively: the
    # OOF frame is seeded with empty lists and built by pd.concat, which can
    # widen the column to float, and a float index is not a valid subscript.
    idx = oof["row_index"].to_numpy().astype(np.int64)
    blend = oof["blend_prob"].to_numpy(dtype=np.float64)

    lookup: Dict[Tuple[str, str], float] = {}
    for row, p in zip(idx, blend):
        lookup[(s1_ids[row], s2_s3_ids[row])] = p

    grouped: Dict[str, List[Tuple[str, float]]] = {}
    for (s1_id, cand_id), p in lookup.items():
        grouped.setdefault(s1_id, []).append((cand_id, float(p)))

    return {
        s1_id: sorted(cands, key=lambda t: (-t[1], t[0]))
        for s1_id, cands in grouped.items()
    }
