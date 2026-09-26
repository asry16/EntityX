"""
Evaluation metrics for Amazon ML Challenge 2026: Business Entity Resolution.
Implements the exact macro-averaged F_0.5 score with singleton handling.
"""

from typing import Dict, Set, Iterable


def compute_entity_f05(true_ids: Set[str], pred_ids: Set[str]) -> float:
    """Compute F_0.5 score for a single Source 1 entity.
    
    Handles singletons (empty true_ids):
    - True empty & Pred empty: 1.0
    - True empty & Pred non-empty: 0.0
    - True non-empty & Pred empty: 0.0
    """
    if not true_ids:
        return 1.0 if not pred_ids else 0.0
    
    if not pred_ids:
        return 0.0
    
    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0
    
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    
    # F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Dict[str, float]:
    """Compute overall evaluation metrics across all Source 1 entities.
    
    Args:
        ground_truth: Dict mapping source1_id -> set of true matching IDs
        predictions: Dict mapping source1_id -> set of predicted matching IDs
        
    Returns:
        Dict with 'macro_f05', 'singleton_accuracy', 'non_singleton_f05',
        'mean_precision', 'mean_recall', 'total_entities', 'total_singletons'
    """
    total_f05 = 0.0
    singleton_correct = 0
    total_singletons = 0
    non_singleton_f05 = 0.0
    non_singleton_count = 0
    sum_prec = 0.0
    sum_rec = 0.0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(true_set, pred_set)
        total_f05 += score
        
        if not true_set:
            total_singletons += 1
            if not pred_set:
                singleton_correct += 1
        else:
            non_singleton_count += 1
            non_singleton_f05 += score
            if pred_set:
                tp = len(true_set & pred_set)
                sum_prec += tp / len(pred_set)
                sum_rec += tp / len(true_set)

    n = len(ground_truth)
    return {
        "macro_f05": total_f05 / n if n > 0 else 0.0,
        "singleton_accuracy": singleton_correct / total_singletons if total_singletons > 0 else 1.0,
        "non_singleton_f05": non_singleton_f05 / non_singleton_count if non_singleton_count > 0 else 0.0,
        "mean_precision": sum_prec / non_singleton_count if non_singleton_count > 0 else 0.0,
        "mean_recall": sum_rec / non_singleton_count if non_singleton_count > 0 else 0.0,
        "total_entities": n,
        "total_singletons": total_singletons,
    }


def evaluate_blocking(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]]
) -> Dict[str, float]:
    """Evaluate candidate generation / blocking stage quality.
    
    Calculates recall ceiling (maximum possible recall of the candidate set)
    and average candidate count per entity.
    """
    total_true_matches = 0
    captured_true_matches = 0
    total_candidates = 0
    entities_with_all_matches_retained = 0
    non_singleton_count = 0

    for s1_id, true_set in ground_truth.items():
        cand_set = candidates.get(s1_id, set())
        total_candidates += len(cand_set)
        
        if true_set:
            non_singleton_count += 1
            total_true_matches += len(true_set)
            captured = len(true_set & cand_set)
            captured_true_matches += captured
            if captured == len(true_set):
                entities_with_all_matches_retained += 1

    n = len(ground_truth)
    return {
        "recall_ceiling": captured_true_matches / total_true_matches if total_true_matches > 0 else 1.0,
        "avg_candidates_per_entity": total_candidates / n if n > 0 else 0.0,
        "full_recall_entities_pct": (entities_with_all_matches_retained / non_singleton_count * 100) if non_singleton_count > 0 else 100.0,
        "total_candidates": total_candidates,
        "total_true_matches": total_true_matches,
        "captured_true_matches": captured_true_matches,
    }
