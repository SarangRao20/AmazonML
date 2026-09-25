"""
Phase 3 orchestrator: Per-country models + routing (Tasks 3, 4, 5).

Coordinates:
- Task 3: Country detection and routing
- Task 4: Country-specific threshold optimization  
- Task 5: Per-country model ensemble creation
"""

from typing import Dict, Set, Tuple, Optional
import pandas as pd
import numpy as np

from .country_analysis import CountryAnalyzer
from .country_models import CountryModelManager
from .country_router import CountryRouter
from .config import PHASE, VERBOSE


class Phase3Orchestrator:
    """Orchestrates Phase 3 per-country model training and inference."""
    
    def __init__(self, verbose: bool = VERBOSE):
        """Initialize orchestrator."""
        self.verbose = verbose
        self.analyzer = CountryAnalyzer(verbose=verbose)
        self.manager = CountryModelManager(verbose=verbose)
        self.router = CountryRouter(country_manager=self.manager, verbose=verbose)
    
    def analyze_and_prepare(self, s1_df: pd.DataFrame, s2_df: pd.DataFrame,
                           s3_df: pd.DataFrame) -> Dict:
        """
        Analyze country distribution and prepare training groups.
        
        Args:
            s1_df: Source 1
            s2_df: Source 2
            s3_df: Source 3
        
        Returns:
            Analysis dict with country groups and error patterns
        """
        if self.verbose:
            print("\n" + "="*80)
            print("PHASE 3: PER-COUNTRY MODEL ANALYSIS & PREPARATION")
            print("="*80)
        
        # Analyze distribution
        country_dist = self.analyzer.analyze_distribution(s1_df, s2_df, s3_df)
        
        # Estimate error patterns
        error_patterns = self.analyzer.estimate_error_patterns(country_dist)
        
        # Create country groups
        country_groups = self.analyzer.create_country_groups(country_dist)
        
        if self.verbose:
            print(f"\n📊 Country Groups:")
            print(f"  High-volume (>10k): {country_groups['high_volume']}")
            print(f"  Medium-volume (1k-10k): {country_groups['medium_volume']}")
            print(f"  Low-volume (<1k): {country_groups['low_volume']}")
            print(f"  Multilingual: {country_groups['multilingual']}")
        
        return {
            'country_dist': country_dist,
            'error_patterns': error_patterns,
            'country_groups': country_groups,
        }
    
    def train_country_models(self, features_df: pd.DataFrame,
                            ground_truth: Dict[str, Set[str]],
                            s1_df: pd.DataFrame,
                            analysis: Dict) -> None:
        """
        Task 5: Train per-country model ensemble.
        
        Args:
            features_df: Feature dataframe
            ground_truth: Ground truth matches
            s1_df: Source 1 entities
            analysis: Output from analyze_and_prepare
        """
        if self.verbose:
            print("\n" + "="*80)
            print("TASK 5: TRAINING PER-COUNTRY MODEL ENSEMBLE")
            print("="*80)
        
        country_groups = analysis['country_groups']
        
        # Partition by country
        country_features = self.manager.partition_by_country(features_df, s1_df)
        
        # Train models for high-volume countries
        trained = 0
        for country in country_groups['high_volume']:
            if country in country_features:
                country_df = country_features[country]
                if len(country_df) > 100:  # Minimum samples
                    self.manager.train_country_model(country, country_df, ground_truth, s1_df)
                    trained += 1
        
        if self.verbose:
            print(f"\n✓ Trained {trained} country-specific models")
            print(f"  Models: {list(self.manager.country_models.keys())}")
    
    def optimize_country_thresholds(self, features_df: pd.DataFrame,
                                   ground_truth: Dict[str, Set[str]],
                                   s1_df: pd.DataFrame,
                                   analysis: Dict) -> None:
        """
        Task 4: Country-specific threshold optimization.
        
        Args:
            features_df: Feature dataframe
            ground_truth: Ground truth
            s1_df: Source 1 entities
            analysis: Analysis dict
        """
        if self.verbose:
            print("\n" + "="*80)
            print("TASK 4: COUNTRY-SPECIFIC THRESHOLD OPTIMIZATION")
            print("="*80)
        
        country_features = self.manager.partition_by_country(features_df, s1_df)
        
        # Optimize thresholds for each trained country model
        optimized = 0
        for country, model in self.manager.country_models.items():
            if country in country_features:
                country_df = country_features[country]
                self.manager.optimize_country_thresholds(country, country_df, ground_truth)
                optimized += 1
        
        if self.verbose:
            print(f"\n✓ Optimized thresholds for {optimized} countries")
            for country, (score_tau, margin_tau) in self.manager.country_thresholds.items():
                print(f"  {country}: score_τ={score_tau:.3f}, margin_τ={margin_tau:.3f}")
    
    def apply_routing(self, features_df: pd.DataFrame,
                     s1_df: pd.DataFrame,
                     global_predictions: np.ndarray,
                     global_confidences: np.ndarray) -> Tuple[Dict[str, Set[str]], Dict]:
        """
        Task 3: Apply country detection and routing.
        
        Args:
            features_df: Features
            s1_df: Source 1
            global_predictions: Global model probabilities
            global_confidences: Global model confidences
        
        Returns:
            (country_routed_predictions, routing_info)
        """
        if self.verbose:
            print("\n" + "="*80)
            print("TASK 3: COUNTRY DETECTION & ROUTING")
            print("="*80)
        
        # Route predictions
        combined_probs, routing_info = self.router.route_predictions(
            features_df, s1_df, global_predictions
        )
        
        # Apply country-specific thresholds
        predictions = self.router.apply_country_thresholds(
            features_df, s1_df, combined_probs, global_confidences
        )
        
        if self.verbose:
            print(f"\n✓ Routing complete:")
            print(f"  Total samples routed: {routing_info['total_samples']:,}")
            print(f"  Via country models: {routing_info['country_model_count']:,}")
            print(f"  Via global fallback: {routing_info['global_fallback_count']:,}")
        
        return predictions, routing_info
    
    def get_phase3_improvement_estimate(self, analysis: Dict) -> float:
        """
        Estimate expected F₀.₅ improvement from Phase 3.
        
        Args:
            analysis: Analysis dict with error patterns
        
        Returns:
            Expected improvement in F₀.₅ points
        """
        error_patterns = analysis['error_patterns']
        
        # Estimate improvement
        # Per-country models typically add 0.5-1.0% F₀.₅
        improvement = 0.0
        
        # High-volume countries (better calibrated models)
        high_volume = sum(1 for p in error_patterns.values() if p['s1_count'] > 10000)
        improvement += 0.4 * min(high_volume / max(len(error_patterns), 1), 1.0)
        
        # Multilingual countries (special handling)
        multilingual = sum(1 for p in error_patterns.values() if p['multilingual'])
        improvement += 0.2 * (multilingual / max(len(error_patterns), 1))
        
        # High recall error countries (better country-specific thresholds)
        high_recall_error = sum(1 for p in error_patterns.values() if p['recall_error'] == 'HIGH')
        improvement += 0.3 * (high_recall_error / max(len(error_patterns), 1))
        
        return min(improvement, 1.0)  # Cap at 1.0%
    
    def save_phase3_models(self) -> None:
        """Save all Phase 3 models."""
        self.manager.save_country_models()
        if self.verbose:
            print("✓ Phase 3 models saved")
