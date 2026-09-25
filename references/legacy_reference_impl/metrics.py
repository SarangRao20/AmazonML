"""Evaluation metrics for Business Entity Resolution Challenge.

Calculates the macro-averaged F_0.5 score across all Source 1 entities:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    = (1.25 * TP) / (1.25 * TP + FP + 0.25 * FN)

Singletons:
- True singleton predicting empty list: score = 1.0
- True singleton predicting any match: score = 0.0
- Non-singleton predicting empty list: score = 0.0
"""

from typing import Dict, Set


def compute_entity_f05(y_true: Set[str], y_pred: Set[str]) -> float:
    """Compute F_0.5 for a single Source 1 entity."""
    if not y_true:
        # Ground truth is a singleton (no matches)
        return 1.0 if not y_pred else 0.0
    
    if not y_pred:
        # Ground truth has matches, but prediction is empty
        return 0.0
    
    tp = len(y_true & y_pred)
    if tp == 0:
        return 0.0
    
    fp = len(y_pred - y_true)
    fn = len(y_true - y_pred)
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
    
    return (1.25 * precision * recall) / denom


def evaluate_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Dict[str, float]:
    """Compute macro-averaged F_0.5 over all entities in ground truth.
    
    Parameters:
        ground_truth: mapping from S1 entity_id to set of true matching IDs
        predictions: mapping from S1 entity_id to set of predicted matching IDs
        
    Returns:
        dict with macro_f05, precision, recall, singleton_acc, non_singleton_f05
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {"macro_f05": 0.0}
    
    scores = []
    singletons_correct = 0
    total_singletons = 0
    non_singleton_scores = []
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(true_set, pred_set)
        scores.append(score)
        
        if not true_set:
            total_singletons += 1
            if not pred_set:
                singletons_correct += 1
        else:
            non_singleton_scores.append(score)
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
    macro_f05 = sum(scores) / total_entities
    singleton_acc = (singletons_correct / total_singletons) if total_singletons > 0 else 1.0
    non_singleton_f05 = (sum(non_singleton_scores) / len(non_singleton_scores)) if non_singleton_scores else 0.0
    
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    return {
        "macro_f05": macro_f05,
        "non_singleton_f05": non_singleton_f05,
        "singleton_acc": singleton_acc,
        "total_entities": total_entities,
        "singletons": total_singletons,
        "micro_precision": micro_p,
        "micro_recall": micro_r
    }
