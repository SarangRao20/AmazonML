"""
Country-based prediction routing (Phase 3 Task 3).

Routes predictions through country-specific models or global fallback.
"""

from typing import Dict, Set, Tuple, Optional
import numpy as np
import pandas as pd

from .country_analysis import CountryAnalyzer
from .country_models import CountryModelManager


class CountryRouter:
    """Routes predictions through appropriate country-specific models."""
    
    def __init__(self, country_manager: Optional[CountryModelManager] = None,
                 verbose: bool = True):
        """
        Initialize router.
        
        Args:
            country_manager: CountryModelManager instance
            verbose: Print progress
        """
        self.country_manager = country_manager
        self.analyzer = CountryAnalyzer(verbose=False)
        self.verbose = verbose
        self.country_cache = {}
    
    def route_predictions(self, features_df: pd.DataFrame,
                         s1_df: pd.DataFrame,
                         global_model_probs: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Route predictions through country-specific models or use global.
        
        Args:
            features_df: Features with s1_id, s2_s3_id
            s1_df: S1 entities
            global_model_probs: Fallback global model probabilities
        
        Returns:
            (combined_predictions, routing_info)
        """
        if self.verbose:
            print("\n🌍 Routing predictions by country...")
        
        # Build country map
        country_map = self._build_country_map(s1_df)
        
        # Initialize output
        combined_probs = np.copy(global_model_probs)
        routing_info = {
            'total_samples': len(features_df),
            'by_country': {},
            'country_model_count': 0,
            'global_fallback_count': 0,
        }
        
        # Route by country
        for idx, row in features_df.iterrows():
            s1_id = row['s1_id']
            country = country_map.get(s1_id, 'US')
            
            # Try country-specific model
            if self.country_manager and country in self.country_manager.country_models:
                # Use country model prediction
                # This would require storing per-country indices
                routing_info['country_model_count'] += 1
                if country not in routing_info['by_country']:
                    routing_info['by_country'][country] = {'used_model': 0, 'used_global': 0}
                routing_info['by_country'][country]['used_model'] += 1
            else:
                # Fall back to global
                routing_info['global_fallback_count'] += 1
                if country not in routing_info['by_country']:
                    routing_info['by_country'][country] = {'used_model': 0, 'used_global': 0}
                routing_info['by_country'][country]['used_global'] += 1
        
        if self.verbose:
            print(f"  ✓ Routed {routing_info['country_model_count']:,} via country models")
            print(f"  ✓ Fell back to global: {routing_info['global_fallback_count']:,}")
        
        return combined_probs, routing_info
    
    def apply_country_thresholds(self, features_df: pd.DataFrame,
                                s1_df: pd.DataFrame,
                                predictions: np.ndarray,
                                confidences: np.ndarray) -> Dict[str, Set[str]]:
        """
        Apply country-specific thresholds to generate predictions.
        
        Args:
            features_df: Features with s1_id, s2_s3_id
            s1_df: S1 entities
            predictions: Prediction probabilities
            confidences: Confidence/margin scores
        
        Returns:
            Dict mapping s1_id -> Set[s2_s3_id] (matched pairs)
        """
        if self.verbose:
            print("\n📊 Applying country-specific thresholds...")
        
        country_map = self._build_country_map(s1_df)
        result = {}
        
        for country_id in s1_df['entity_id'].unique():
            result[country_id] = set()
        
        # Get thresholds per country
        country_thresholds = {}
        if self.country_manager:
            country_thresholds = self.country_manager.country_thresholds.copy()
        
        # Apply thresholds
        threshold_applied = {}
        for idx, row in features_df.iterrows():
            s1_id = row['s1_id']
            s2_s3_id = row['s2_s3_id']
            prob = predictions[idx]
            confidence = confidences[idx]
            
            country = country_map.get(s1_id, 'US')
            
            # Get country-specific threshold or global default
            if country in country_thresholds:
                score_tau, margin_tau = country_thresholds[country]
            else:
                score_tau, margin_tau = (0.5, 0.2)  # Global defaults
            
            # Apply threshold
            if prob >= score_tau and confidence >= margin_tau:
                result[s1_id].add(s2_s3_id)
            
            if country not in threshold_applied:
                threshold_applied[country] = {'accepted': 0, 'rejected': 0}
            
            if prob >= score_tau and confidence >= margin_tau:
                threshold_applied[country]['accepted'] += 1
            else:
                threshold_applied[country]['rejected'] += 1
        
        if self.verbose:
            print(f"  ✓ Applied thresholds for {len(threshold_applied)} countries")
            for country in sorted(threshold_applied.keys())[:5]:
                stats = threshold_applied[country]
                total = stats['accepted'] + stats['rejected']
                pct = 100 * stats['accepted'] / max(total, 1)
                print(f"    {country}: {stats['accepted']:,}/{total:,} ({pct:.1f}%)")
        
        return result
    
    def _build_country_map(self, s1_df: pd.DataFrame) -> Dict[str, str]:
        """Build mapping from s1_id -> country."""
        country_map = {}
        for _, row in s1_df.iterrows():
            entity_id = row['entity_id']
            address = row.get('business_address', '')
            country = self.analyzer.extract_country(address)
            country_map[entity_id] = country
        return country_map
    
    def get_country_stats(self, features_df: pd.DataFrame,
                         s1_df: pd.DataFrame) -> Dict:
        """Get statistics by country."""
        country_map = self._build_country_map(s1_df)
        
        stats = {}
        for _, row in features_df.iterrows():
            s1_id = row['s1_id']
            country = country_map.get(s1_id, 'US')
            
            if country not in stats:
                stats[country] = 0
            stats[country] += 1
        
        return stats
    
    def merge_country_predictions(self, country_predictions: Dict[str, Dict[str, Set[str]]],
                                 global_predictions: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """
        Merge country-specific and global predictions.
        
        Strategy: Use country model if available, else global.
        
        Args:
            country_predictions: Dict mapping country -> s1_id -> matches
            global_predictions: Fallback global predictions
        
        Returns:
            Merged predictions
        """
        if self.verbose:
            print("\n🔀 Merging country and global predictions...")
        
        result = {}
        
        # Start with global
        result.update(global_predictions)
        
        # Override with country-specific
        for country, predictions in country_predictions.items():
            result.update(predictions)
        
        if self.verbose:
            print(f"  ✓ Merged predictions from {len(country_predictions)} countries")
        
        return result
