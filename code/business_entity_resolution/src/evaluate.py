"""
Evaluation metrics and analysis for entity resolution.

Metrics:
- Macro F₀.₅ (official leaderboard metric)
- Per-entity F₀.₅ (entity-level breakdown)
- Precision, Recall (per-entity)
- Singleton accuracy
- Per-country performance (US, India, France)
- Per-source-pair (S1-S2 vs S1-S3)

Error analysis to identify improvement opportunities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple
from collections import defaultdict

from .config import F_BETA, VERBOSE


class MetricsComputer:
    """Compute entity resolution metrics."""
    
    @staticmethod
    def compute_entity_metrics(y_true: Set[str], y_pred: Set[str],
                              beta: float = 0.5) -> Dict[str, float]:
        """
        Compute precision, recall, F_β for a single entity.
        
        Args:
            y_true: Set of true matched IDs
            y_pred: Set of predicted matched IDs
            beta: Beta parameter for F_β
        
        Returns:
            Dict with precision, recall, f_beta
        """
        # Handle singletons
        if not y_true and not y_pred:
            return {
                'precision': 1.0,
                'recall': 1.0,
                'f_beta': 1.0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
            }
        if not y_true or not y_pred:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f_beta': 0.0,
                'tp': 0,
                'fp': len(y_pred),
                'fn': len(y_true),
            }
        
        # Compute TP, FP, FN
        tp = len(y_true & y_pred)
        fp = len(y_pred - y_true)
        fn = len(y_true - y_pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall == 0:
            f_beta = 0.0
        else:
            beta_sq = beta ** 2
            f_beta = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f_beta': f_beta,
            'tp': tp,
            'fp': fp,
            'fn': fn,
        }
    
    @staticmethod
    def compute_macro_metrics(predictions: Dict[str, Set[str]],
                             ground_truth: Dict[str, Set[str]],
                             beta: float = 0.5) -> Dict[str, float]:
        """
        Compute macro-averaged metrics across all entities.
        
        Args:
            predictions: Dict mapping entity ID -> set of predicted matches
            ground_truth: Dict mapping entity ID -> set of true matches
            beta: Beta parameter
        
        Returns:
            Dict with macro-averaged metrics
        """
        all_metrics = []
        
        for s1_id in ground_truth.keys():
            y_true = ground_truth[s1_id]
            y_pred = predictions.get(s1_id, set())
            
            metrics = MetricsComputer.compute_entity_metrics(y_true, y_pred, beta=beta)
            all_metrics.append(metrics)
        
        # Macro average
        macro_metrics = {
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'recall': np.mean([m['recall'] for m in all_metrics]),
            'f_beta': np.mean([m['f_beta'] for m in all_metrics]),
        }
        
        return macro_metrics
    
    @staticmethod
    def compute_per_entity_metrics(predictions: Dict[str, Set[str]],
                                  ground_truth: Dict[str, Set[str]],
                                  s1_records: pd.DataFrame,
                                  beta: float = 0.5) -> pd.DataFrame:
        """
        Compute per-entity metrics with country and entity info.
        
        Args:
            predictions: Dict of predictions
            ground_truth: Dict of ground truth
            s1_records: S1 dataframe with metadata
            beta: Beta parameter
        
        Returns:
            DataFrame with per-entity metrics
        """
        results = []
        
        # Create mapping of entity ID -> metadata
        s1_meta = {row['entity_id']: row.to_dict() for _, row in s1_records.iterrows()}
        
        for s1_id in ground_truth.keys():
            y_true = ground_truth[s1_id]
            y_pred = predictions.get(s1_id, set())
            
            metrics = MetricsComputer.compute_entity_metrics(y_true, y_pred, beta=beta)
            
            # Get metadata
            meta = s1_meta.get(s1_id, {})
            
            results.append({
                'entity_id': s1_id,
                'country': meta.get('country', 'unknown'),
                'true_matches': len(y_true),
                'pred_matches': len(y_pred),
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f_beta': metrics['f_beta'],
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'is_singleton': len(y_true) == 0,
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def compute_country_breakdown(predictions: Dict[str, Set[str]],
                                 ground_truth: Dict[str, Set[str]],
                                 s1_records: pd.DataFrame,
                                 beta: float = 0.5) -> Dict[str, Dict]:
        """
        Compute metrics breakdown by country.
        
        Args:
            predictions: Dict of predictions
            ground_truth: Dict of ground truth
            s1_records: S1 dataframe
            beta: Beta parameter
        
        Returns:
            Dict mapping country -> metrics
        """
        # Get per-entity metrics
        per_entity = MetricsComputer.compute_per_entity_metrics(
            predictions, ground_truth, s1_records, beta=beta
        )
        
        country_metrics = {}
        
        for country in per_entity['country'].unique():
            country_data = per_entity[per_entity['country'] == country]
            
            country_metrics[country] = {
                'count': len(country_data),
                'precision': country_data['precision'].mean(),
                'recall': country_data['recall'].mean(),
                'f_beta': country_data['f_beta'].mean(),
                'singletons': country_data['is_singleton'].sum(),
                'singleton_accuracy': country_data[country_data['is_singleton']]['f_beta'].mean(),
            }
        
        return country_metrics
    
    @staticmethod
    def compute_source_pair_breakdown(predictions: Dict[str, Set[str]],
                                     ground_truth: Dict[str, Set[str]],
                                     s2_df: pd.DataFrame,
                                     s3_df: pd.DataFrame,
                                     beta: float = 0.5) -> Dict[str, Dict]:
        """
        Compute metrics breakdown by source pair (S1-S2 vs S1-S3).
        
        Args:
            predictions: Dict of predictions
            ground_truth: Dict of ground truth
            s2_df: Source 2 dataframe
            s3_df: Source 3 dataframe
            beta: Beta parameter
        
        Returns:
            Dict mapping source_pair -> metrics
        """
        s2_ids = set(s2_df['entity_id'].unique())
        s3_ids = set(s3_df['entity_id'].unique())
        
        s1_s2_metrics = []
        s1_s3_metrics = []
        
        for s1_id in ground_truth.keys():
            y_true = ground_truth[s1_id]
            y_pred = predictions.get(s1_id, set())
            
            # Split by source
            y_true_s2 = {m for m in y_true if m in s2_ids}
            y_pred_s2 = {m for m in y_pred if m in s2_ids}
            
            y_true_s3 = {m for m in y_true if m in s3_ids}
            y_pred_s3 = {m for m in y_pred if m in s3_ids}
            
            s1_s2_metrics.append(MetricsComputer.compute_entity_metrics(y_true_s2, y_pred_s2, beta=beta))
            s1_s3_metrics.append(MetricsComputer.compute_entity_metrics(y_true_s3, y_pred_s3, beta=beta))
        
        return {
            'S1-S2': {
                'precision': np.mean([m['precision'] for m in s1_s2_metrics]),
                'recall': np.mean([m['recall'] for m in s1_s2_metrics]),
                'f_beta': np.mean([m['f_beta'] for m in s1_s2_metrics]),
                'samples': len(s1_s2_metrics),
            },
            'S1-S3': {
                'precision': np.mean([m['precision'] for m in s1_s3_metrics]),
                'recall': np.mean([m['recall'] for m in s1_s3_metrics]),
                'f_beta': np.mean([m['f_beta'] for m in s1_s3_metrics]),
                'samples': len(s1_s3_metrics),
            },
        }
    
    @staticmethod
    def identify_error_patterns(per_entity_metrics: pd.DataFrame,
                               predictions: Dict[str, Set[str]],
                               ground_truth: Dict[str, Set[str]],
                               top_n: int = 10) -> Dict[str, List]:
        """
        Identify error patterns for improvement (Phase 2 enhanced).
        
        Categorizes errors into:
        - False positives (FP): Predicted matches that shouldn't be
        - False negatives (FN): Missed matches
        - Precision errors: Too many predicted matches
        - Recall errors: Missed some correct matches
        - Edge cases: Single entity with many errors
        
        Args:
            per_entity_metrics: Per-entity metrics dataframe
            predictions: Predictions
            ground_truth: Ground truth
            top_n: Top N errors to report
        
        Returns:
            Dict with categorized error patterns
        """
        errors = {
            'false_positives': [],
            'false_negatives': [],
            'precision_errors': [],
            'recall_errors': [],
            'worst_f_beta': [],
            'edge_cases': [],
        }
        
        # False positives (predicted match but shouldn't be)
        fp_entities = per_entity_metrics[per_entity_metrics['fp'] > 0].nlargest(top_n, 'fp')
        errors['false_positives'] = [
            {
                'entity_id': row['entity_id'],
                'fp_count': int(row['fp']),
                'tp_count': int(row['tp']),
                'error_rate': row['fp'] / (row['tp'] + row['fp']) if (row['tp'] + row['fp']) > 0 else 1.0,
                'precision': row['precision'],
            }
            for _, row in fp_entities.iterrows()
        ]
        
        # False negatives (missed matches)
        fn_entities = per_entity_metrics[per_entity_metrics['fn'] > 0].nlargest(top_n, 'fn')
        errors['false_negatives'] = [
            {
                'entity_id': row['entity_id'],
                'fn_count': int(row['fn']),
                'tp_count': int(row['tp']),
                'error_rate': row['fn'] / (row['tp'] + row['fn']) if (row['tp'] + row['fn']) > 0 else 1.0,
                'recall': row['recall'],
            }
            for _, row in fn_entities.iterrows()
        ]
        
        # Precision errors (low precision entities)
        precision_errors = per_entity_metrics[per_entity_metrics['precision'] < 0.5].nsmallest(top_n, 'precision')
        errors['precision_errors'] = [
            {
                'entity_id': row['entity_id'],
                'precision': row['precision'],
                'fp': int(row['fp']),
                'tp': int(row['tp']),
                'total_predicted': int(row['tp'] + row['fp']),
            }
            for _, row in precision_errors.iterrows()
        ]
        
        # Recall errors (low recall entities)
        recall_errors = per_entity_metrics[per_entity_metrics['recall'] < 0.5].nsmallest(top_n, 'recall')
        errors['recall_errors'] = [
            {
                'entity_id': row['entity_id'],
                'recall': row['recall'],
                'fn': int(row['fn']),
                'tp': int(row['tp']),
                'total_true': int(row['tp'] + row['fn']),
            }
            for _, row in recall_errors.iterrows()
        ]
        
        # Worst F₀.₅ scores
        worst_f_entities = per_entity_metrics.nsmallest(top_n, 'f_beta')
        errors['worst_f_beta'] = [
            {
                'entity_id': row['entity_id'],
                'f_beta': row['f_beta'],
                'precision': row['precision'],
                'recall': row['recall'],
                'tp': int(row['tp']),
                'fp': int(row['fp']),
                'fn': int(row['fn']),
            }
            for _, row in worst_f_entities.iterrows()
        ]
        
        # Edge cases: Entities with anomalous error patterns
        # High FP/TP ratio OR high FN/TP ratio
        per_entity_metrics['fp_tp_ratio'] = per_entity_metrics['fp'] / (per_entity_metrics['tp'] + 1)
        per_entity_metrics['fn_tp_ratio'] = per_entity_metrics['fn'] / (per_entity_metrics['tp'] + 1)
        
        edge_case_candidates = per_entity_metrics[
            (per_entity_metrics['fp_tp_ratio'] > 2.0) | (per_entity_metrics['fn_tp_ratio'] > 2.0)
        ].nlargest(top_n, 'fp_tp_ratio')
        
        errors['edge_cases'] = [
            {
                'entity_id': row['entity_id'],
                'fp_tp_ratio': row['fp_tp_ratio'],
                'fn_tp_ratio': row['fn_tp_ratio'],
                'anomaly_type': 'high_fp' if row['fp_tp_ratio'] > row['fn_tp_ratio'] else 'high_fn',
            }
            for _, row in edge_case_candidates.iterrows()
        ]
        
        return errors


def evaluate_predictions(predictions: Dict[str, Set[str]],
                        ground_truth: Dict[str, Set[str]],
                        s1_records: pd.DataFrame,
                        s2_df: pd.DataFrame,
                        s3_df: pd.DataFrame,
                        verbose: bool = True) -> Dict:
    """
    Comprehensive evaluation of predictions.
    
    Args:
        predictions: Dict of predictions
        ground_truth: Dict of ground truth
        s1_records: S1 dataframe
        s2_df: Source 2 dataframe
        s3_df: Source 3 dataframe
        verbose: Print results
    
    Returns:
        Dict with all evaluation metrics
    """
    computer = MetricsComputer()
    
    # Macro metrics
    macro = computer.compute_macro_metrics(predictions, ground_truth, beta=F_BETA)
    
    # Per-entity metrics
    per_entity = computer.compute_per_entity_metrics(predictions, ground_truth, s1_records, beta=F_BETA)
    
    # Country breakdown
    country_breakdown = computer.compute_country_breakdown(predictions, ground_truth, s1_records, beta=F_BETA)
    
    # Source pair breakdown
    source_breakdown = computer.compute_source_pair_breakdown(predictions, ground_truth, s2_df, s3_df, beta=F_BETA)
    
    # Error patterns
    errors = computer.identify_error_patterns(per_entity, predictions, ground_truth)
    
    if verbose:
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n📊 MACRO METRICS:")
        print(f"   Precision: {macro['precision']*100:.2f}%")
        print(f"   Recall: {macro['recall']*100:.2f}%")
        print(f"   F₀.₅: {macro['f_beta']*100:.2f}%")
        
        print(f"\n🌍 BY COUNTRY:")
        for country, metrics in country_breakdown.items():
            print(f"   {country}:")
            print(f"      Count: {metrics['count']:,}")
            print(f"      F₀.₅: {metrics['f_beta']*100:.2f}%")
            print(f"      Singletons: {metrics['singletons']} ({metrics['singleton_accuracy']*100:.1f}% correct)")
        
        print(f"\n📍 BY SOURCE PAIR:")
        for pair, metrics in source_breakdown.items():
            print(f"   {pair}: F₀.₅ = {metrics['f_beta']*100:.2f}%")
        
        print(f"\n⚠️  TOP ERRORS:")
        print(f"   Top False Positives:")
        for err in errors['false_positives'][:3]:
            print(f"      {err['entity_id']}: {err['fp_count']} FP ({err['error_rate']*100:.1f}% error)")
        
        print(f"   Top False Negatives:")
        for err in errors['false_negatives'][:3]:
            print(f"      {err['entity_id']}: {err['fn_count']} FN ({err['error_rate']*100:.1f}% error)")
    
    return {
        'macro': macro,
        'per_entity': per_entity,
        'country_breakdown': country_breakdown,
        'source_breakdown': source_breakdown,
        'errors': errors,
    }


if __name__ == "__main__":
    # Test evaluation
    print("=" * 80)
    print("Testing MetricsComputer")
    print("=" * 80)
    
    # Synthetic data
    predictions = {
        'S1-001': {'S2-010', 'S3-020'},
        'S1-002': {'S2-030'},
        'S1-003': set(),
        'S1-004': {'S2-040', 'S2-050', 'S2-055'},  # False positive: S2-055
    }
    
    ground_truth = {
        'S1-001': {'S2-010', 'S3-020'},
        'S1-002': {'S2-030'},
        'S1-003': set(),
        'S1-004': {'S2-040', 'S2-050'},  # Missing S2-051
    }
    
    computer = MetricsComputer()
    macro = computer.compute_macro_metrics(predictions, ground_truth, beta=0.5)
    
    print(f"\nMacro Metrics:")
    print(f"  Precision: {macro['precision']*100:.2f}%")
    print(f"  Recall: {macro['recall']*100:.2f}%")
    print(f"  F₀.₅: {macro['f_beta']*100:.2f}%")
    
    print("\n✅ Evaluation test completed!")
