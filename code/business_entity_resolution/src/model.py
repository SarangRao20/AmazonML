"""
Model training with tri-ensemble (XGBoost + LightGBM + CatBoost).

Architecture:
- 5-Fold GroupKFold cross-validation (strict entity-level disjoint)
- Class imbalance weighting: scale_pos_weight = sqrt(neg/pos)
- Three diverse gradient boosting models blended: XGB (40%) + LGB (35%) + CatBoost (25%)
- Calibration: Isotonic regression on validation fold
- Out-of-fold (OOF) predictions for threshold tuning

Expected improvement: +0.7-1.0 F₀.₅ points from ensemble diversity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import pickle
from pathlib import Path

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression

from .config import (
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS,
    ENSEMBLE_WEIGHTS, CLASS_WEIGHT_STRATEGY, CLASS_WEIGHT_MIN,
    CLASS_WEIGHT_MAX, CLASS_WEIGHT_MULTIPLIER, EARLY_STOPPING_ROUNDS,
    RANDOM_SEED, N_SPLITS_CV, MODELS_DIR, USE_PROBABILITY_CALIBRATION,
    VERBOSE
)


class TriEnsembleModel:
    """Tri-model ensemble for entity matching."""
    
    def __init__(self, verbose: bool = True):
        """Initialize ensemble."""
        self.verbose = verbose
        self.models = {
            "xgboost": None,
            "lightgbm": None,
            "catboost": None,
        }
        self.calibrators = {
            "xgboost": None,
            "lightgbm": None,
            "catboost": None,
        }
        self.feature_names = None
        
    def compute_class_weight(self, y: np.ndarray) -> float:
        """
        Compute class weight for imbalanced data.
        
        Strategy: scale_pos_weight = sqrt(neg/pos)
        - Positive samples: matches (rare, ~6%)
        - Negative samples: non-matches (common, ~94%)
        
        Formula:
        scale_pos_weight = min(MAX, max(MIN, MULT * sqrt(neg/pos)))
        
        Args:
            y: Binary labels (0/1)
        
        Returns:
            Computed scale_pos_weight
        """
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        
        if n_pos == 0:
            weight = CLASS_WEIGHT_MAX
        else:
            weight = np.sqrt(n_neg / n_pos)
            weight = CLASS_WEIGHT_MULTIPLIER * weight
            weight = np.clip(weight, CLASS_WEIGHT_MIN, CLASS_WEIGHT_MAX)
        
        if self.verbose:
            print(f"   Class weight: {weight:.2f} (neg/pos ratio: {n_neg/n_pos:.1f})")
        
        return weight
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: np.ndarray,
                     X_val: pd.DataFrame, y_val: np.ndarray) -> xgb.XGBClassifier:
        """
        Train XGBoost model with class imbalance weighting.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            Trained XGBoost model
        """
        if self.verbose:
            print("  🚀 Training XGBoost...")
        
        scale_weight = self.compute_class_weight(y_train)
        
        model = xgb.XGBClassifier(
            **XGBOOST_PARAMS,
            scale_pos_weight=scale_weight,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        
        # Train with early stopping on validation set
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        if self.verbose:
            print(f"     ✓ Best round: {model.best_iteration}")
        
        return model
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: np.ndarray,
                      X_val: pd.DataFrame, y_val: np.ndarray) -> lgb.LGBMClassifier:
        """
        Train LightGBM model with class imbalance weighting.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            Trained LightGBM model
        """
        if self.verbose:
            print("  🚀 Training LightGBM...")
        
        scale_weight = self.compute_class_weight(y_train)
        
        model = lgb.LGBMClassifier(
            **LIGHTGBM_PARAMS,
            scale_pos_weight=scale_weight,
        )
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=0),
            ],
        )
        
        if self.verbose:
            print(f"     ✓ Trained")
        
        return model
    
    def train_catboost(self, X_train: pd.DataFrame, y_train: np.ndarray,
                      X_val: pd.DataFrame, y_val: np.ndarray) -> CatBoostClassifier:
        """
        Train CatBoost model with class imbalance weighting.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            Trained CatBoost model
        """
        if self.verbose:
            print("  🚀 Training CatBoost...")
        
        scale_weight = self.compute_class_weight(y_train)
        
        model = CatBoostClassifier(
            **CATBOOST_PARAMS,
            scale_pos_weight=scale_weight,
        )
        
        # Train
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False,
        )
        
        if self.verbose:
            print(f"     ✓ Trained")
        
        return model
    
    def calibrate_predictions(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                             model_name: str) -> IsotonicRegression:
        """
        Fit isotonic regression calibrator on validation fold.
        
        Calibration ensures probabilities are well-calibrated [0,1] without saturation.
        Critical for precise threshold tuning.
        
        Args:
            y_true: True labels
            y_pred_proba: Raw model probabilities
            model_name: Name for logging
        
        Returns:
            Fitted IsotonicRegression calibrator
        """
        if self.verbose:
            print(f"     Calibrating {model_name}...")
        
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_pred_proba, y_true)
        
        return calibrator
    
    def train_groupkfold(self, X: pd.DataFrame, y: np.ndarray,
                        groups: np.ndarray) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Train ensemble with 5-Fold GroupKFold cross-validation.
        
        Entity-level disjoint split:
        - All pairs of the same S1 entity stay together (train or val fold)
        - Zero entity overlap between folds
        - Out-of-fold (OOF) predictions for validation
        
        Args:
            X: Features dataframe
            y: Binary labels
            groups: S1 entity IDs (for grouping)
        
        Returns:
            Tuple of (trained_models_dict, oof_predictions_df)
        """
        if self.verbose:
            print(f"\n📊 Training with {N_SPLITS_CV}-Fold GroupKFold...")
        
        gkf = GroupKFold(n_splits=N_SPLITS_CV)
        
        oof_predictions = pd.DataFrame({
            'fold': [],
            'xgb_prob': [],
            'lgb_prob': [],
            'catboost_prob': [],
            'blend_prob': [],
            'true_label': [],
        })
        
        trained_models = {
            'fold_models': [],  # List of fold-specific models
            'full_models': {},   # Models trained on full data
            'calibrators': {},   # Calibrators per model
        }
        
        fold_idx = 0
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            fold_idx += 1
            if self.verbose:
                print(f"\n  Fold {fold_idx}/{N_SPLITS_CV}:")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Remove ID columns if present
            X_train_clean = X_train.drop(columns=['s1_id', 's2_s3_id'], errors='ignore')
            X_val_clean = X_val.drop(columns=['s1_id', 's2_s3_id'], errors='ignore')
            
            # Store feature names
            self.feature_names = X_train_clean.columns.tolist()
            
            if self.verbose:
                print(f"     Train: {len(X_train):,} pairs, Val: {len(X_val):,} pairs")
            
            # Train three models
            xgb_model = self.train_xgboost(X_train_clean, y_train, X_val_clean, y_val)
            lgb_model = self.train_lightgbm(X_train_clean, y_train, X_val_clean, y_val)
            catboost_model = self.train_catboost(X_train_clean, y_train, X_val_clean, y_val)
            
            # Get validation predictions
            xgb_pred = xgb_model.predict_proba(X_val_clean)[:, 1]
            lgb_pred = lgb_model.predict_proba(X_val_clean)[:, 1]
            catboost_pred = catboost_model.predict_proba(X_val_clean)[:, 1]
            
            # Calibrate on this fold
            if USE_PROBABILITY_CALIBRATION:
                xgb_calibrator = self.calibrate_predictions(y_val, xgb_pred, "XGBoost")
                lgb_calibrator = self.calibrate_predictions(y_val, lgb_pred, "LightGBM")
                catboost_calibrator = self.calibrate_predictions(y_val, catboost_pred, "CatBoost")
                
                xgb_pred_calib = xgb_calibrator.predict(xgb_pred)
                lgb_pred_calib = lgb_calibrator.predict(lgb_pred)
                catboost_pred_calib = catboost_calibrator.predict(catboost_pred)
            else:
                xgb_pred_calib = xgb_pred
                lgb_pred_calib = lgb_pred
                catboost_pred_calib = catboost_pred
                xgb_calibrator = lgb_calibrator = catboost_calibrator = None
            
            # Blend predictions
            blend_pred = (
                ENSEMBLE_WEIGHTS['xgboost'] * xgb_pred_calib +
                ENSEMBLE_WEIGHTS['lightgbm'] * lgb_pred_calib +
                ENSEMBLE_WEIGHTS['catboost'] * catboost_pred_calib
            )
            
            # Store fold results. Carry the original feature-row index so the
            # caller can realign OOF predictions with the feature matrix:
            # pd.concat below yields fold order, not input order.
            fold_oof = pd.DataFrame({
                'row_index': X_val.index.to_numpy(),
                'fold': [fold_idx] * len(X_val),
                'xgb_prob': xgb_pred_calib,
                'lgb_prob': lgb_pred_calib,
                'catboost_prob': catboost_pred_calib,
                'blend_prob': blend_pred,
                'true_label': y_val,
            }, index=X_val.index)
            
            oof_predictions = pd.concat([oof_predictions, fold_oof], ignore_index=True)
            
            trained_models['fold_models'].append({
                'fold': fold_idx,
                'xgb': xgb_model,
                'lgb': lgb_model,
                'catboost': catboost_model,
                'xgb_calib': xgb_calibrator,
                'lgb_calib': lgb_calibrator,
                'catboost_calib': catboost_calibrator,
            })
        
        # Train on full data for production
        if self.verbose:
            print(f"\n  Training on full data for production...")
        
        X_clean = X.drop(columns=['s1_id', 's2_s3_id'], errors='ignore')
        
        full_xgb = self.train_xgboost(X_clean, y, X_clean[:100], y[:100])
        full_lgb = self.train_lightgbm(X_clean, y, X_clean[:100], y[:100])
        full_catboost = self.train_catboost(X_clean, y, X_clean[:100], y[:100])
        
        trained_models['full_models'] = {
            'xgb': full_xgb,
            'lgb': full_lgb,
            'catboost': full_catboost,
        }
        
        if self.verbose:
            print(f"\n✅ Training complete! OOF predictions collected: {len(oof_predictions):,}")
        
        return trained_models, oof_predictions
    
    def predict_ensemble(self, X: pd.DataFrame, models: Dict[str, Any],
                        use_calibration: bool = True) -> np.ndarray:
        """
        Make ensemble predictions on new data.
        
        Args:
            X: Features dataframe
            models: Trained models dict (from train_groupkfold)
            use_calibration: Whether to apply calibration
        
        Returns:
            Blended probability predictions
        """
        X_clean = X.drop(columns=['s1_id', 's2_s3_id'], errors='ignore')
        
        full_models = models.get('full_models', {})
        
        # Get predictions from each model
        xgb_pred = full_models['xgb'].predict_proba(X_clean)[:, 1]
        lgb_pred = full_models['lgb'].predict_proba(X_clean)[:, 1]
        catboost_pred = full_models['catboost'].predict_proba(X_clean)[:, 1]
        
        # Blend
        blend = (
            ENSEMBLE_WEIGHTS['xgboost'] * xgb_pred +
            ENSEMBLE_WEIGHTS['lightgbm'] * lgb_pred +
            ENSEMBLE_WEIGHTS['catboost'] * catboost_pred
        )
        
        return blend
    
    def save_models(self, models: Dict[str, Any], path: Path = MODELS_DIR):
        """
        Save trained models to disk.
        
        Args:
            models: Trained models dict
            path: Directory to save
        """
        if self.verbose:
            print(f"\n💾 Saving models to {path}...")
        
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        
        # Save full models
        with open(path / "xgboost_model.pkl", "wb") as f:
            pickle.dump(models['full_models']['xgb'], f)
        
        with open(path / "lightgbm_model.pkl", "wb") as f:
            pickle.dump(models['full_models']['lgb'], f)
        
        with open(path / "catboost_model.pkl", "wb") as f:
            pickle.dump(models['full_models']['catboost'], f)
        
        if self.verbose:
            print(f"✅ Models saved!")
    
    def load_models(self, path: Path = MODELS_DIR) -> Dict[str, Any]:
        """
        Load trained models from disk.
        
        Args:
            path: Directory to load from
        
        Returns:
            Loaded models dict
        """
        if self.verbose:
            print(f"\n📥 Loading models from {path}...")
        
        path = Path(path)
        
        with open(path / "xgboost_model.pkl", "rb") as f:
            xgb_model = pickle.load(f)
        
        with open(path / "lightgbm_model.pkl", "rb") as f:
            lgb_model = pickle.load(f)
        
        with open(path / "catboost_model.pkl", "rb") as f:
            catboost_model = pickle.load(f)
        
        return {
            'full_models': {
                'xgb': xgb_model,
                'lgb': lgb_model,
                'catboost': catboost_model,
            }
        }


if __name__ == "__main__":
    # Test model training
    print("=" * 80)
    print("Testing TriEnsembleModel")
    print("=" * 80)
    
    from .data_loader import DataLoader
    from .blocking import create_blocking_pipeline
    from .features import extract_features_from_candidates
    
    # Load data
    loader = DataLoader(use_val_sample=True, verbose=True)
    s1, s2, s3, gt = loader.load_data()
    gt_dict = loader.build_ground_truth_dict()
    
    # Block
    print("\n" + "="*80)
    candidates, recall = create_blocking_pipeline(s1, s2, s3, gt_dict, verbose=True)
    print(f"Recall: {recall*100:.2f}%")
    
    # Extract features
    print("\n" + "="*80)
    features_df = extract_features_from_candidates(candidates, s1, s2, s3, verbose=True)
    
    # Add labels
    print("\n" + "="*80)
    s2_s3_records = {row['entity_id']: row.to_dict() for _, row in pd.concat([s2, s3]).iterrows()}
    gt_dict_with_s2s3 = {}
    for s1_id, s1_row in s1.iterrows():
        s1_entity_id = s1_row['entity_id']
        if s1_entity_id in gt_dict:
            gt_dict_with_s2s3[s1_entity_id] = gt_dict[s1_entity_id]
    
    labels = []
    for _, row in features_df.iterrows():
        s1_id = row['s1_id']
        s2_s3_id = row['s2_s3_id']
        label = 1 if s2_s3_id in gt_dict_with_s2s3.get(s1_id, set()) else 0
        labels.append(label)
    
    features_df['label'] = labels
    
    # Get groups for GroupKFold
    s1_ids = s1['entity_id'].values
    groups = features_df['s1_id'].map({s1_id: i for i, s1_id in enumerate(s1_ids)}).values
    
    # Train ensemble
    print("\n" + "="*80)
    ensemble = TriEnsembleModel(verbose=True)
    X = features_df.drop(columns=['label'], errors='ignore')
    y = features_df['label'].values
    
    trained_models, oof_preds = ensemble.train_groupkfold(X, y, groups)
    
    print("\n✅ Model training test completed!")
