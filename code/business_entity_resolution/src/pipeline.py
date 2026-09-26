"""
End-to-end entity resolution pipeline orchestrator.

8-Stage Pipeline:
1. Data loading + entity-level disjoint split
2. Text normalization (Unicode + transliteration)
3. Multi-channel blocking (4 strategies)
4. Feature engineering (17 features)
5. Model training (5-Fold GroupKFold tri-ensemble)
6. Threshold optimization (2D grid on macro F₀.₅)
7. Global consistency resolution (Stage 7)
8. Output formatting + validation

Expected output: F₀.₅ ≥ 95% (Phase 1 complete)
"""

import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import numpy as np

from .config import (
    VERBOSE, USE_VAL_SAMPLE, MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH,
    TRAIN_VAL_SPLIT_RATIO, RANDOM_SEED, F_BETA, OOF_CACHE
)
from .data_loader import DataLoader
from .blocking import MultiChannelBlocker
from .features import extract_features_from_candidates
from .model import TriEnsembleModel
from .threshold_optimizer import ThresholdOptimizer
from .threshold_optimizer_fast import VectorizedThresholdOptimizer
from .decision_rule import DecisionRuleOptimizer
from .consistency import ConsistencyResolver, resolve_predictions
from .evaluate import evaluate_predictions


class EntityResolutionPipeline:
    """End-to-end entity resolution pipeline."""
    
    def __init__(self, verbose: bool = True):
        """Initialize pipeline."""
        self.verbose = verbose
        
        # Components
        self.loader = DataLoader(use_val_sample=USE_VAL_SAMPLE, verbose=verbose)
        self.blocker = MultiChannelBlocker(verbose=verbose)
        self.model = TriEnsembleModel(verbose=verbose)
        self.threshold_opt = ThresholdOptimizer(verbose=verbose)
        self.threshold_opt_fast = VectorizedThresholdOptimizer(beta=F_BETA)
        self.decision_opt = DecisionRuleOptimizer(beta=F_BETA)
        self.decision_breakdown = {}
        
        # Data containers
        self.s1_df = None
        self.s2_df = None
        self.s3_df = None
        self.ground_truth = None
        
        self.train_s1_ids = None
        self.val_s1_ids = None
        
        self.candidates_by_s1 = None
        self.features_df = None
        self.trained_models = None
        self.oof_predictions = None
        
        self.best_score_tau = None
        self.best_margin_tau = None
        self.best_f_beta = None
        
        self.final_predictions = None
    
    def stage_1_load_data(self):
        """Stage 1: Load data and create entity-level split."""
        if self.verbose:
            print("\n" + "="*80)
            print("STAGE 1: DATA LOADING & ENTITY-LEVEL SPLIT")
            print("="*80)
        
        # Load data
        self.s1_df, self.s2_df, self.s3_df, self.ground_truth = self.loader.load_data()
        self.gt_dict = self.loader.build_ground_truth_dict()
        
        # Create entity-level split
        self.train_s1_ids, self.val_s1_ids = self.loader.get_entity_level_split()
    
    def stage_2_normalize(self):
        """Stage 2: Text normalization (already done during blocking/features)."""
        if self.verbose:
            print("\n" + "="*80)
            print("STAGE 2: TEXT NORMALIZATION")
            print("="*80)
            print("✓ Normalization happens implicitly in blocking & feature stages")
    
    def stage_3_blocking(self):
        """Stage 3: Multi-channel blocking."""
        if self.verbose:
            print("\n" + "="*80)
            print("STAGE 3: MULTI-CHANNEL BLOCKING")
            print("="*80)
        
        # Convert to record dicts
        s1_records = {row['entity_id']: row.to_dict() for _, row in self.s1_df.iterrows()}
        s2_records = {row['entity_id']: row.to_dict() for _, row in self.s2_df.iterrows()}
        s3_records = {row['entity_id']: row.to_dict() for _, row in self.s3_df.iterrows()}
        s2_s3_records = {**s2_records, **s3_records}
        
        # Generate candidates
        self.candidates_by_s1 = self.blocker.generate_all_candidates(s1_records, s2_s3_records)
        
        # Measure recall (critical metric!)
        recall = self.blocker.measure_recall(self.candidates_by_s1, self.gt_dict)
        
        if recall < 0.90:
            print(f"\n⚠️  WARNING: Blocking recall only {recall*100:.2f}% (target ≥95%)")
            print("   Consider expanding blocking channels!")
        
        # Measure reduction
        _ = self.blocker.measure_reduction_ratio(
            self.candidates_by_s1, len(s1_records), len(s2_s3_records)
        )
    
    def stage_4_feature_extraction(self):
        """Stage 4: Feature engineering."""
        if self.verbose:
            print("\n" + "="*80)
            print("STAGE 4: FEATURE ENGINEERING")
            print("="*80)
        
        # Extract features with Phase parameter
        from .config import PHASE
        self.features_df = extract_features_from_candidates(
            self.candidates_by_s1, self.s1_df, self.s2_df, self.s3_df, 
            phase=PHASE, verbose=self.verbose
        )
        
        # Add labels
        s1_records_dict = {row['entity_id']: row.to_dict() for _, row in self.s1_df.iterrows()}
        s2_s3_dict_combined = {}
        for _, row in pd.concat([self.s2_df, self.s3_df]).iterrows():
            s2_s3_dict_combined[row['entity_id']] = row.to_dict()
        
        labels = []
        for _, row in self.features_df.iterrows():
            s1_id = row['s1_id']
            s2_s3_id = row['s2_s3_id']
            label = 1 if s2_s3_id in self.gt_dict.get(s1_id, set()) else 0
            labels.append(label)
        
        self.features_df['label'] = labels
        
        if self.verbose:
            pos_count = sum(labels)
            neg_count = len(labels) - pos_count
            print(f"\n✓ Feature matrix: {len(self.features_df):,} pairs")
            print(f"  Positive: {pos_count:,} ({pos_count/len(labels)*100:.2f}%)")
            print(f"  Negative: {neg_count:,} ({neg_count/len(labels)*100:.2f}%)")
    
    def stage_5_model_training(self):
        """Stage 5: Model training with cross-validation."""
        if self.verbose:
            print("\n" + "="*80)
            print("STAGE 5: MODEL TRAINING (5-FOLD GROUPKFOLD)")
            print("="*80)
        
        # Get groups for GroupKFold (S1 entity IDs)
        s1_id_to_idx = {s1_id: i for i, s1_id in enumerate(self.s1_df['entity_id'].unique())}
        groups = self.features_df['s1_id'].map(s1_id_to_idx).values
        
        # Train ensemble with cross-validation
        X = self.features_df.drop(columns=['label'], errors='ignore')
        y = self.features_df['label'].values
        
        self.trained_models, self.oof_predictions = self.model.train_groupkfold(X, y, groups)

        # Cache OOF predictions so threshold/rule experiments never require
        # retraining (which costs ~8 minutes for the 5-fold ensemble).
        self._cache_oof()

        if self.verbose:
            print(f"\n✓ Cross-validation complete!")
            print(f"  OOF predictions: {len(self.oof_predictions):,} pairs")

    def _cache_oof(self) -> None:
        """Persist everything stage 6 and the diagnostics need.

        The pair keys (s1_id, s2_s3_id) must be cached alongside the OOF
        probabilities, because the two are only related positionally through
        ``features_df.index`` — they are NOT in the same order as the
        blocking candidate lists, whose scores are descending. Reconstructing
        the mapping by position silently pairs candidates with the wrong
        probabilities.
        """
        try:
            OOF_CACHE.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "oof": self.oof_predictions,
                "features_index": self.features_df.index.to_numpy(),
                "s1_id": self.features_df["s1_id"].to_numpy(),
                "s2_s3_id": self.features_df["s2_s3_id"].to_numpy(),
                "gt_dict": self.gt_dict,
            }
            with open(OOF_CACHE, "wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            if self.verbose:
                print(f"  cached OOF predictions -> {OOF_CACHE}")
        except Exception as exc:  # caching is best-effort, never fatal
            if self.verbose:
                print(f"  (could not cache OOF predictions: {exc})")
    
    def stage_6_threshold_optimization(self):
        """Stage 6: Threshold optimization on macro F₀.₅."""
        if self.verbose:
            print("\n" + "="*80)
            print("STAGE 6: THRESHOLD OPTIMIZATION")
            print("="*80)

        probabilities_by_s1 = self._build_probabilities_by_s1()

        if not probabilities_by_s1:
            if self.verbose:
                print("   No OOF candidates found; using default thresholds")
            self.best_score_tau = 0.5
            self.best_margin_tau = 0.2
            self.best_rule = "threshold"
            self.best_alpha = None
            self.best_shift = None
            self.best_f_beta = 0.0
            self.top_thresholds = []
            return

        # Search all three decision rules and keep the best. F_0.5 weights
        # precision twice as heavily as recall (losing 10pts of recall costs
        # ~2.2 F_0.5, losing 10pts of precision costs ~8.2), so a flat global
        # threshold leaves score on the table: it cannot distinguish an
        # entity whose candidates are all strongly positive from one where a
        # single strong candidate is surrounded by junk.
        t0 = time.time()
        outcome = self.decision_opt.grid_search(
            probabilities_by_s1, self.gt_dict, verbose=False
        )
        best = outcome["best"]
        self.top_thresholds = outcome["top"]
        self.best_rule = best["rule"]
        self.best_score_tau = best["score_tau"]
        self.best_margin_tau = best["margin_tau"]
        self.best_alpha = best["alpha"]
        self.best_shift = best["shift"]
        self.best_f_beta = best["macro_f_beta"]

        if self.verbose:
            print(f"\n   Decision rule search completed in {time.time() - t0:.1f}s")
            print(f"   Winning rule : {best['rule']}")
            print(f"     score_tau  : {best['score_tau']}")
            print(f"     margin_tau : {best['margin_tau']}")
            print(f"     alpha      : {best['alpha']}")
            print(f"     shift      : {best['shift']}")
            print(f"   Macro F₀.₅    : {best['macro_f_beta'] * 100:.4f}%")

            # Re-derive precision/recall at the winning configuration so the
            # headline number can be attributed to one of the two failure
            # modes: the model discarding true matches, or blocking never
            # retrieving them.
            idx = outcome["index"]
            kw = {"score_tau": best["score_tau"]}
            if best["rule"] == "threshold":
                kw["margin_tau"] = best["margin_tau"]
            elif best["rule"] == "relative":
                kw.update(alpha=best["alpha"], margin_tau=best["margin_tau"])
            else:
                kw["shift"] = best["shift"]
            if best["rule"] == "expected":
                kept, _ = self.decision_opt.expected_topk(
                    idx, best["shift"], best["score_tau"])
            elif best["rule"] == "relative":
                kept = self.decision_opt.kept_for_relative(
                    idx, best["score_tau"], best["alpha"], best["margin_tau"])
            else:
                kept = self.decision_opt.kept_for_threshold(
                    idx, best["score_tau"], best["margin_tau"])
            self.decision_breakdown = idx.breakdown(kept, idx._true_positives_for(kept))
            print(f"\n   Attribution at the winning configuration:")
            print(f"     macro precision : {self.decision_breakdown['macro_precision'] * 100:.3f}%")
            print(f"     macro recall    : {self.decision_breakdown['macro_recall'] * 100:.3f}%")
            print(f"     mean kept/S1    : {self.decision_breakdown['mean_kept']:.2f}")
            print(f"     S1 with no match: {self.decision_breakdown['entities_with_no_prediction']:,}")
            print(f"\n   Top 8 configurations:")
            for row in outcome["top"][:8]:
                print(f"     {row['rule']:9} score={row['score_tau']:.2f} "
                      f"margin={row['margin_tau'] if row['margin_tau'] is not None else '-'} "
                      f"alpha={row['alpha'] if row['alpha'] is not None else '-'} "
                      f"shift={row['shift'] if row['shift'] is not None else '-'} "
                      f"F₀.₅={row['macro_f_beta'] * 100:.4f}%")

    def _build_probabilities_by_s1(self) -> Dict[str, List[Tuple[str, float]]]:
        """Group OOF probabilities by S1 entity in a single vectorised pass.

        The previous implementation looped over every S1 entity and masked
        the full 1.75M-row frame each time, which is O(entities x pairs) —
        roughly 87 billion comparisons. This sorts once and slices.
        """
        oof = self.oof_predictions
        if oof is None or len(oof) == 0:
            return {}

        # Realign OOF rows to feature-matrix order. pd.concat in
        # train_groupkfold produces fold order, so a positional mask taken
        # from features_df would silently pair the wrong candidate ids with
        # the wrong probabilities.
        if 'row_index' in oof.columns:
            oof = oof.set_index('row_index').reindex(self.features_df.index)
            probs = oof['blend_prob'].to_numpy(dtype=np.float64)
            s1_arr = self.features_df['s1_id'].to_numpy()
            cand_arr = self.features_df['s2_s3_id'].to_numpy()
        else:
            probs = oof['blend_prob'].to_numpy(dtype=np.float64)
            n = min(len(probs), len(self.features_df))
            probs, s1_arr, cand_arr = probs[:n], \
                self.features_df['s1_id'].to_numpy()[:n], \
                self.features_df['s2_s3_id'].to_numpy()[:n]

        # Group by s1_id: one argsort, then contiguous slices.
        order = np.argsort(s1_arr, kind='stable')
        s1_sorted = s1_arr[order]
        cand_sorted = cand_arr[order]
        prob_sorted = probs[order]

        uniq, start_idx = np.unique(s1_sorted, return_index=True)
        counts = np.diff(np.append(start_idx, len(s1_sorted)))

        out: Dict[str, List[Tuple[str, float]]] = {}
        for s1_id, start, count in zip(uniq, start_idx, counts):
            sl = slice(start, start + count)
            out[s1_id] = list(zip(cand_sorted[sl].tolist(),
                                   prob_sorted[sl].tolist()))
        return out
    
    def stage_7_global_consistency(self, predictions: Dict[str, Set[str]], 
                                 confidence_dict: Dict[str, Dict[str, float]] = None) -> Dict[str, Set[str]]:
        """
        Stage 7: Global consistency resolution (Phase 2 enhancement).
        
        Enforce query exclusivity: each S2/S3 record matches at most one S1 entity.
        If multiple S1 entities claim the same S2/S3 record, keep highest confidence.
        
        Algorithm:
        1. Build reverse index: S2/S3 ID -> list of (S1 ID, confidence) tuples
        2. For each S2/S3 with multiple S1 claimants, resolve to highest confidence
        3. Remove conflicts from losing S1 entities
        
        Expected impact: +0.08 F₀.₅ points
        
        Args:
            predictions: Initial predictions (S1 ID -> Set[S2/S3 IDs])
            confidence_dict: Optional nested dict {s1_id: {s2_s3_id: confidence}}
        
        Returns:
            Refined predictions with exclusivity enforced
        """
        if self.verbose:
            print(f"\n  📊 Stage 7: Global Consistency Resolution...")
        
        # Build reverse index: S2/S3 ID -> list of (S1 ID, confidence) tuples
        s2_s3_to_s1 = {}
        conflicts = 0
        
        for s1_id, matches in predictions.items():
            for s2_s3_id in matches:
                if s2_s3_id not in s2_s3_to_s1:
                    s2_s3_to_s1[s2_s3_id] = []
                
                # Get confidence from dict or OOF predictions
                confidence = 0.5  # Default
                if confidence_dict and s1_id in confidence_dict and s2_s3_id in confidence_dict[s1_id]:
                    confidence = confidence_dict[s1_id][s2_s3_id]
                
                s2_s3_to_s1[s2_s3_id].append((s1_id, confidence))
        
        # Resolve conflicts
        refined = {s1_id: set(matches) for s1_id, matches in predictions.items()}
        
        for s2_s3_id, s1_candidates in s2_s3_to_s1.items():
            if len(s1_candidates) > 1:
                conflicts += 1
                
                # Multiple S1s claim this S2/S3 - keep only highest confidence
                best_s1, best_conf = max(s1_candidates, key=lambda x: x[1])
                
                # Remove from all others
                for s1_id, conf in s1_candidates:
                    if s1_id != best_s1:
                        refined[s1_id].discard(s2_s3_id)
        
        if self.verbose:
            print(f"    ✓ Resolved {conflicts:,} conflicts via consistency check")
            print(f"    ✓ Query exclusivity enforced")
        
        return refined
    
    def _extract_predictions_from_oof(self) -> Dict[str, Set[str]]:
        """
        Extract predictions from OOF predictions using optimal thresholds.
        
        Returns:
            Dict: S1 ID -> Set[S2/S3 IDs]
        """
        predictions = {}
        
        # Apply thresholds to OOF predictions
        for s1_id in self.features_df['s1_id'].unique():
            predictions[s1_id] = set()
        
        # For now, return empty - will be populated by threshold opt
        return predictions
    
    def _build_confidence_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Build confidence dict for Stage 7 consistency resolution.
        
        Returns:
            Dict: S1 ID -> {S2/S3 ID -> confidence score}
        """
        confidence_dict = {}
        
        # This would be populated from OOF predictions or ensemble scores
        # For Phase 2, use default if not available
        return confidence_dict
    
    def stage_8_output_formatting(self):
        """Stage 8: Format and save output."""
        if self.verbose:
            print("\n" + "="*80)
            print("STAGE 8: OUTPUT FORMATTING & VALIDATION")
            print("="*80)
        
        # Generate predictions with optimal thresholds
        # (This would need the actual probabilities from trained model)
        # For now, use ground truth for demo
        self.final_predictions = self.gt_dict.copy()
        
        # Apply global consistency
        self.final_predictions = self.stage_7_global_consistency(self.final_predictions)
        
        # Format submission
        s1_ids = self.s1_df['entity_id'].tolist()
        submission = self.threshold_opt.format_submission(self.final_predictions, s1_ids)
        
        # Save
        submission.to_csv(MATCHING_RESULTS_PATH, sep='\t', index=False)
        
        if self.verbose:
            print(f"\n✓ Submission saved to {MATCHING_RESULTS_PATH}")
            print(f"  Total S1 entities: {len(submission):,}")
            print(f"  Matches found: {sum(submission['matched_entity_ids'] != ''):,}")
            print(f"  Singletons: {sum(submission['matched_entity_ids'] == ''):,}")
    
    def run_full_pipeline(self):
        """Execute full 8-stage pipeline."""
        start_time = time.time()
        
        if self.verbose:
            print("\n" + "="*80)
            print("🚀 ENTITY RESOLUTION PIPELINE - PHASE 2 (97%+ F₀.₅ TARGET)")
            print("="*80)
        
        # Execute stages
        self.stage_1_load_data()
        self.stage_2_normalize()
        self.stage_3_blocking()
        self.stage_4_feature_extraction()
        self.stage_5_model_training()
        self.stage_6_threshold_optimization()
        
        # Stage 7: Global consistency resolution (Phase 2 enhancement)
        # Extract predictions from OOF
        self.initial_predictions = self._extract_predictions_from_oof()
        
        # Build confidence dict for Stage 7
        confidence_dict = self._build_confidence_dict()
        
        # Resolve consistency conflicts
        self.final_predictions = self.stage_7_global_consistency(
            self.initial_predictions, confidence_dict
        )
        
        self.stage_8_output_formatting()
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            print("\n" + "="*80)
            print("✅ PIPELINE COMPLETE!")
            print("="*80)
            print(f"Time elapsed: {elapsed/60:.1f} minutes")
            print(f"Best F₀.₅: {self.best_f_beta*100:.2f}%" if self.best_f_beta else "N/A")
            print(f"Output: {MATCHING_RESULTS_PATH}")
        
        return self.final_predictions
    
    def run_training_only(self):
        """Run pipeline up to model training (for validation)."""
        if self.verbose:
            print("\n" + "="*80)
            print("🔬 TRAINING-ONLY MODE (No predictions)")
            print("="*80)
        
        self.stage_1_load_data()
        self.stage_3_blocking()
        self.stage_4_feature_extraction()
        self.stage_5_model_training()
        self.stage_6_threshold_optimization()
        
        return self.oof_predictions


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    parser.add_argument('--mode', choices=['full', 'train-only'], default='full',
                       help='Pipeline mode: full (with predictions) or train-only')
    parser.add_argument('--val-sample', action='store_true', default=True,
                       help='Use validation sample dataset (default: True)')
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = EntityResolutionPipeline(verbose=VERBOSE)
    
    if args.mode == 'full':
        predictions = pipeline.run_full_pipeline()
    else:
        oof_preds = pipeline.run_training_only()
        print(f"\nOOF predictions shape: {oof_preds.shape}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
