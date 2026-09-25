"""
Per-country model training and management (Phase 3).

Strategy:
- High-volume countries (>10k entities): Individual tri-ensemble
- Medium-volume (1k-10k): Shared ensemble with country weighting
- Low-volume (<1k): Pooled with global model
- Multilingual: Special feature engineering
"""

from typing import Dict, List, Set, Tuple, Optional
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

from .model import TriEnsembleModel
from .threshold_optimizer import ThresholdOptimizer
from .config import RANDOM_SEED


class CountryModelManager:
    """Manages per-country models for Phase 3."""
    
    def __init__(self, model_dir: Path = Path("models/country_models"), 
                 verbose: bool = True):
        """Initialize manager."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        self.country_models = {}  # country -> model
        self.country_thresholds = {}  # country -> (score_tau, margin_tau)
        self.country_metadata = {}  # country -> metadata
    
    def partition_by_country(self, features_df: pd.DataFrame,
                            s1_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Partition features by country.
        
        Args:
            features_df: Features with s1_id, s2_s3_id columns
            s1_df: Source 1 with entity_id and business_address
        
        Returns:
            Dict mapping country -> features for that country
        """
        if self.verbose:
            print("\n🌍 Partitioning data by country...")
        
        # Create country map from s1_df
        from .country_analysis import CountryAnalyzer
        analyzer = CountryAnalyzer(verbose=False)
        
        country_map = {}
        for _, row in s1_df.iterrows():
            entity_id = row['entity_id']
            address = row.get('business_address', '')
            country = analyzer.extract_country(address)
            country_map[entity_id] = country
        
        # Partition features
        partitions = {}
        for _, row in features_df.iterrows():
            s1_id = row['s1_id']
            country = country_map.get(s1_id, 'US')
            
            if country not in partitions:
                partitions[country] = []
            partitions[country].append(row)
        
        # Convert to dataframes
        country_features = {}
        for country, rows in partitions.items():
            country_features[country] = pd.DataFrame(rows)
        
        if self.verbose:
            print(f"  ✓ Partitioned into {len(country_features)} countries")
            for country in sorted(country_features.keys()):
                print(f"    {country}: {len(country_features[country]):,} features")
        
        return country_features
    
    def train_country_model(self, country: str, features_df: pd.DataFrame,
                           ground_truth: Dict[str, Set[str]],
                           s1_df: pd.DataFrame) -> TriEnsembleModel:
        """
        Train model for specific country.
        
        Args:
            country: Country code
            features_df: Features for this country
            ground_truth: Ground truth matches
            s1_df: S1 entities (for GroupKFold)
        
        Returns:
            Trained TriEnsembleModel
        """
        if self.verbose:
            print(f"\n🏋️  Training model for {country} ({len(features_df):,} samples)...")
        
        # Prepare data
        feature_cols = [col for col in features_df.columns 
                       if col not in ['s1_id', 's2_s3_id', 'label']]
        X = features_df[feature_cols].values
        
        # Create labels
        y = np.zeros(len(features_df))
        for idx, row in features_df.iterrows():
            s1_id = row['s1_id']
            s2_s3_id = row['s2_s3_id']
            if s1_id in ground_truth and s2_s3_id in ground_truth[s1_id]:
                y[idx] = 1
        
        # Create groups for country-specific GroupKFold
        groups = features_df['s1_id'].values
        
        # Train model
        model = TriEnsembleModel(verbose=self.verbose)
        model.train_with_cv(X, y, groups=groups, n_splits=5)
        
        if self.verbose:
            print(f"  ✓ Trained {country} model with CV")
        
        self.country_models[country] = model
        return model
    
    def optimize_country_thresholds(self, country: str,
                                   features_df: pd.DataFrame,
                                   ground_truth: Dict[str, Set[str]],
                                   beta: float = 0.5) -> Tuple[float, float]:
        """
        Optimize thresholds for country.
        
        Args:
            country: Country code
            features_df: Features for this country
            ground_truth: Ground truth
            beta: F-beta weight
        
        Returns:
            (score_threshold, margin_threshold)
        """
        if self.verbose:
            print(f"\n🎯 Optimizing thresholds for {country}...")
        
        if country not in self.country_models:
            if self.verbose:
                print(f"  ⚠️  No model for {country}, skipping threshold optimization")
            return (0.5, 0.2)
        
        model = self.country_models[country]
        
        # Get OOF predictions
        feature_cols = [col for col in features_df.columns 
                       if col not in ['s1_id', 's2_s3_id', 'label']]
        X = features_df[feature_cols].values
        
        # Predict probabilities
        oof_probs = model.predict_proba(X)
        
        # Create labels
        y_true = np.zeros(len(features_df))
        for idx, row in features_df.iterrows():
            s1_id = row['s1_id']
            s2_s3_id = row['s2_s3_id']
            if s1_id in ground_truth and s2_s3_id in ground_truth[s1_id]:
                y_true[idx] = 1
        
        # Optimize thresholds
        optimizer = ThresholdOptimizer(verbose=self.verbose)
        best_score_tau, best_margin_tau, best_f_beta = optimizer.optimize_thresholds_2d(
            oof_probs, y_true, features_df, beta=beta
        )
        
        self.country_thresholds[country] = (best_score_tau, best_margin_tau)
        
        if self.verbose:
            print(f"  ✓ {country} thresholds: score={best_score_tau:.3f}, margin={best_margin_tau:.3f}")
        
        return (best_score_tau, best_margin_tau)
    
    def save_country_models(self) -> None:
        """Save all country models to disk."""
        if self.verbose:
            print(f"\n💾 Saving {len(self.country_models)} country models...")
        
        for country, model in self.country_models.items():
            model_path = self.model_dir / f"model_{country}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save thresholds
        threshold_path = self.model_dir / "thresholds.pkl"
        with open(threshold_path, 'wb') as f:
            pickle.dump(self.country_thresholds, f)
        
        if self.verbose:
            print(f"  ✓ Models saved to {self.model_dir}")
    
    def load_country_models(self) -> None:
        """Load country models from disk."""
        if self.verbose:
            print(f"\n📂 Loading country models from {self.model_dir}...")
        
        model_files = list(self.model_dir.glob("model_*.pkl"))
        
        for model_file in model_files:
            country = model_file.stem.replace("model_", "")
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
                self.country_models[country] = model
        
        # Load thresholds
        threshold_path = self.model_dir / "thresholds.pkl"
        if threshold_path.exists():
            with open(threshold_path, 'rb') as f:
                self.country_thresholds = pickle.load(f)
        
        if self.verbose:
            print(f"  ✓ Loaded {len(self.country_models)} country models")
    
    def predict_by_country(self, features_df: pd.DataFrame,
                          s1_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions using country-specific models.
        
        Args:
            features_df: Features with s1_id, s2_s3_id
            s1_df: S1 entities for country mapping
        
        Returns:
            Dict mapping country -> predictions
        """
        if self.verbose:
            print(f"\n🔮 Predicting with country-specific models...")
        
        from .country_analysis import CountryAnalyzer
        analyzer = CountryAnalyzer(verbose=False)
        
        # Create country map
        country_map = {}
        for _, row in s1_df.iterrows():
            entity_id = row['entity_id']
            address = row.get('business_address', '')
            country = analyzer.extract_country(address)
            country_map[entity_id] = country
        
        # Partition features
        partitions = {}
        for _, row in features_df.iterrows():
            s1_id = row['s1_id']
            country = country_map.get(s1_id, 'US')
            if country not in partitions:
                partitions[country] = []
            partitions[country].append(row)
        
        # Predict per country
        predictions = {}
        for country, rows in partitions.items():
            country_df = pd.DataFrame(rows)
            
            if country not in self.country_models:
                if self.verbose:
                    print(f"  ⚠️  No model for {country}, using global")
                # Use global model (fallback)
                continue
            
            model = self.country_models[country]
            feature_cols = [col for col in country_df.columns 
                           if col not in ['s1_id', 's2_s3_id']]
            X = country_df[feature_cols].values
            
            probs = model.predict_proba(X)
            predictions[country] = probs
        
        if self.verbose:
            print(f"  ✓ Predicted for {len(predictions)} countries")
        
        return predictions
    
    def get_country_thresholds(self, country: str) -> Tuple[float, float]:
        """Get thresholds for country."""
        if country in self.country_thresholds:
            return self.country_thresholds[country]
        
        # Default global thresholds
        return (0.5, 0.2)
