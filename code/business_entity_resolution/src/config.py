"""
Configuration module for Amazon ML Challenge 2026 Entity Resolution Pipeline.

Centralized configuration for paths, hyperparameters, blocking strategies,
model settings, and threshold optimization ranges.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# ==================== DATASET PATHS ====================
def _find_project_root() -> Path:
    """Locate the repo root by walking up until a ``dataset/`` dir appears.

    Robust to this file being moved between ``code/business_entity_resolution/``
    and ``code/business_entity_resolution/src/``. Override with the
    ``AMZ_ML_ROOT`` environment variable if the layout ever changes.
    """
    override = os.environ.get("AMZ_ML_ROOT")
    if override:
        return Path(override).resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "dataset").is_dir():
            return candidate
    # Fall back to three levels up (repo root in the documented layout).
    return here.parents[2]


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Training data paths
TRAIN_SOURCE1_PATH = DATA_DIR / "train" / "train_source1.tsv"
TRAIN_SOURCE2_PATH = DATA_DIR / "train" / "train_source2.tsv"
TRAIN_SOURCE3_PATH = DATA_DIR / "train" / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH = DATA_DIR / "train" / "train_ground_truth.tsv"

# Test data paths
TEST_SOURCE1_PATH = DATA_DIR / "test" / "test_source1.tsv"
TEST_SOURCE2_PATH = DATA_DIR / "test" / "test_source2.tsv"
TEST_SOURCE3_PATH = DATA_DIR / "test" / "test_source3.tsv"

# Validation sample paths
VAL_SOURCE1_PATH = DATA_DIR / "val_sample" / "sample_source1.tsv"
VAL_SOURCE2_PATH = DATA_DIR / "val_sample" / "sample_source2.tsv"
VAL_SOURCE3_PATH = DATA_DIR / "val_sample" / "sample_source3.tsv"
VAL_GROUND_TRUTH_PATH = DATA_DIR / "val_sample" / "sample_ground_truth.tsv"

# Output submission paths
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Model checkpoint paths
MODELS = {
    "xgboost": MODELS_DIR / "xgboost_model.pkl",
    "lightgbm": MODELS_DIR / "lightgbm_model.pkl",
    "catboost": MODELS_DIR / "catboost_model.pkl",
    "calibrator": MODELS_DIR / "calibrator.pkl",
}

# ==================== SPLITTING CONFIGURATION ====================
# Entity-level disjoint split (GroupKFold)
TRAIN_VAL_SPLIT_RATIO = 0.7  # 70% train, 30% validation
N_SPLITS_CV = 5  # 5-Fold GroupKFold for cross-validation
RANDOM_SEED = 42

# ==================== NORMALIZATION CONFIGURATION ====================
# Text normalization settings
NORMALIZE_UNICODE = True  # NFKC normalization
NORMALIZE_LEGAL_SUFFIX = True
NORMALIZE_TRANSLITERATE = True  # Devanagari -> Latin

# Legal suffixes to expand/strip
LEGAL_SUFFIXES = {
    "corp": "corporation",
    "pvt": "private",
    "ltd": "limited",
    "inc": "incorporated",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "pllc": "professional limited liability company",
}

# ==================== BLOCKING CONFIGURATION ====================
# Multi-channel blocking strategy
BLOCKING_CONFIG = {
    "channel_1": {
        "name": "name_tokens",
        "description": "Name token inverted index (raw + transliterated ASCII)",
        "min_token_length": 3,
        "weight": 1.0,
        "enabled": True,
    },
    "channel_2": {
        "name": "address_tokens",
        "description": "Address token inverted index",
        "min_token_length": 3,
        "weight": 1.0,
        "enabled": True,
    },
    "channel_3": {
        "name": "street_number_street",
        "description": "Street number + street name prefix",
        "min_token_length": 2,
        "weight": 1.5,
        "enabled": True,
    },
    "channel_4": {
        "name": "postal_distinctive",
        "description": "Postal code + distinctive address pairs",
        "min_token_length": 4,
        "weight": 2.0,
        "enabled": True,
    },
}

# Blocking parameters
CANDIDATES_PER_S1 = 35  # Keep top-K candidates per S1 entity
STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "of", "for",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "could", "may", "might",
}

# ==================== FEATURE ENGINEERING CONFIGURATION ====================
# Feature extraction settings (Phase 1: 17 features, Phase 2: 55 features)

# Phase 1 features (17 core)
FEATURE_NAMES_PHASE1 = [
    # Name similarity (5 features)
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    
    # Address similarity (5 features)
    "addr_ratio",
    "addr_partial_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaro_winkler",
    
    # Number overlap (1 feature - Project 2 discovery, +0.74 impact)
    "number_overlap",
    
    # Token-level (2 features)
    "token_overlap_count",
    "token_jaccard_sim",
    
    # Exact match flags (2 features)
    "name_exact_match",
    "addr_exact_match",
    
    # Structural (2 features)
    "name_length_ratio",
    "addr_length_ratio",
]

# Phase 2 additional features (38 new)
FEATURE_NAMES_PHASE2_NEW = [
    # Legal suffix (2 features)
    "both_have_suffix",
    "suffix_match",
    
    # Composite/interaction (7 features)
    "harmonic_mean",
    "product",
    "minimum",
    "maximum",
    "abs_diff",
    "high_both",
    "composite_score",
    
    # Postal hierarchical (4 features)
    "postal_exact",
    "postal_prefix_2",
    "postal_prefix_3",
    "postal_distance",
    
    # Landmark (2 features)
    "landmark_overlap",
    "domain_match",
    
    # Multi-channel meta-features (19 features - Task 4)
    # Per-channel indicators (8)
    "ch1_name_tokens",
    "ch2_addr_tokens",
    "ch3_street_number",
    "ch4_postal",
    "ch1_rank",
    "ch2_rank",
    "ch3_rank",
    "ch4_rank",
    
    # Best-rank features (4)
    "best_channel_rank",
    "rank_agreement",
    "reciprocal_rank",
    "channel_count",
    
    # Agreement features (7)
    "channels_agree",
    "name_addr_agreement",
    "top_channel",
    "channel_diversity",
    "cross_channel_rank",
    "avg_channel_rank",
    "channel_confidence",
]

# Character n-gram features (added alongside the reference-inspired blocking work)
FEATURE_NAMES_PHASE3_NGRAM = [
    "name_char3_jaccard",
    "addr_char3_jaccard",
]

# Full feature set
FEATURE_NAMES = FEATURE_NAMES_PHASE1 + FEATURE_NAMES_PHASE2_NEW + FEATURE_NAMES_PHASE3_NGRAM

NUM_FEATURES = len(FEATURE_NAMES)
NUM_FEATURES_PHASE1 = len(FEATURE_NAMES_PHASE1)
NUM_FEATURES_PHASE2 = len(FEATURE_NAMES_PHASE1) + len(FEATURE_NAMES_PHASE2_NEW)
NUM_FEATURES_NGRAM = len(FEATURE_NAMES_PHASE3_NGRAM)

# Phase 2 breakdown
NUM_FEATURES_LEGAL_SUFFIX = 2
NUM_FEATURES_COMPOSITE = 7
NUM_FEATURES_POSTAL = 4
NUM_FEATURES_LANDMARK = 2
NUM_FEATURES_MULTICHANNEL = 19  # Task 4
NUM_FEATURES_PHASE2_NEW_TOTAL = NUM_FEATURES_LEGAL_SUFFIX + NUM_FEATURES_COMPOSITE + NUM_FEATURES_POSTAL + NUM_FEATURES_LANDMARK + NUM_FEATURES_MULTICHANNEL

assert NUM_FEATURES_PHASE2_NEW_TOTAL == 34, f"Expected 34 Phase 2 new features, got {NUM_FEATURES_PHASE2_NEW_TOTAL}"
assert NUM_FEATURES == 53, f"Expected 53 total features (17 + 34 + 2 n-gram), got {NUM_FEATURES}"

# ==================== PHASE 2 CONFIGURATION ====================
# Phase selection (1 = 17 features, 2 = 55 features)
PHASE = 2

# Feature flags for Phase 2 enhancements
ENABLE_LEGAL_SUFFIX_FEATURES = True  # +2 features
ENABLE_COMPOSITE_FEATURES = True  # +7 features
ENABLE_POSTAL_HIERARCHICAL = True  # +4 features
ENABLE_LANDMARK_FEATURES = True  # +2 features
ENABLE_MULTICHANNEL_FEATURES = True  # +19 features

# Stage 7: Global consistency resolution
ENABLE_GLOBAL_CONSISTENCY = True  # Enforce query exclusivity
CONSISTENCY_RESOLUTION_METHOD = "confidence_based"  # Options: confidence_based, majority_vote

# Embedding-based blocking (optional Phase 4 upgrade)
ENABLE_EMBEDDING_BLOCKING = False  # Set to True if sentence-transformers + faiss installed
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_TOP_K = 50
EMBEDDING_WEIGHT = 0.4
TOKEN_BLOCKING_WEIGHT = 0.6

# ==================== MODEL CONFIGURATION ====================
# Tri-model ensemble weights

# Tri-model ensemble weights
ENSEMBLE_WEIGHTS = {
    "xgboost": 0.40,
    "lightgbm": 0.35,
    "catboost": 0.25,
}

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Phase 2 XGBoost params (can be tuned for 55 features)
XGBOOST_PARAMS_PHASE2 = {
    "n_estimators": 350,  # +50 trees for more features
    "max_depth": 9,  # +1 for feature interaction capacity
    "learning_rate": 0.04,  # Slightly lower for more stable training
    "subsample": 0.75,  # Slightly lower
    "colsample_bytree": 0.75,  # Feature subsampling
    "min_child_weight": 4,  # +1 to reduce overfitting
    "gamma": 1.5,  # Slightly higher regularization
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# LightGBM hyperparameters
LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Phase 2 LightGBM params
LIGHTGBM_PARAMS_PHASE2 = {
    "n_estimators": 350,  # +50 for more features
    "max_depth": 9,  # +1
    "learning_rate": 0.04,  # Slightly lower
    "num_leaves": 40,  # +9 for better splits on 55 features
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "min_child_samples": 6,  # +1
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# CatBoost hyperparameters
CATBOOST_PARAMS = {
    "iterations": 300,
    "depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": RANDOM_SEED,
    "verbose": False,
}

# Phase 2 CatBoost params
CATBOOST_PARAMS_PHASE2 = {
    "iterations": 350,  # +50
    "depth": 9,  # +1
    "learning_rate": 0.04,  # Slightly lower
    "subsample": 0.75,
    "l2_leaf_reg": 5.0,  # L2 regularization for more features
    "random_state": RANDOM_SEED,
    "verbose": False,
}

# Class imbalance weighting
CLASS_WEIGHT_STRATEGY = "sqrt_ratio"  # sqrt(neg/pos)
CLASS_WEIGHT_MIN = 2.0
CLASS_WEIGHT_MAX = 12.0
CLASS_WEIGHT_MULTIPLIER = 1.5

# Early stopping
EARLY_STOPPING_ROUNDS = 50
EARLY_STOPPING_METRIC = "f1"  # Monitor F1 during training

# ==================== THRESHOLD OPTIMIZATION ====================
# 2D grid search on (score_threshold, margin_threshold)
SCORE_THRESHOLD_RANGE = (0.3, 0.95)
SCORE_THRESHOLD_STEP = 0.05

MARGIN_THRESHOLD_RANGE = (0.05, 0.5)
MARGIN_THRESHOLD_STEP = 0.05

# Post-processing
USE_PROBABILITY_CALIBRATION = True
USE_GLOBAL_CONSISTENCY = True  # Stage 7: enforce query exclusivity

# Phase 2 threshold tuning (can be overridden per phase)
PHASE2_SCORE_THRESHOLD_RANGE = (0.35, 0.93)  # Slightly tighter for higher precision
PHASE2_SCORE_THRESHOLD_STEP = 0.04

PHASE2_MARGIN_THRESHOLD_RANGE = (0.08, 0.45)  # Slightly tighter margin
PHASE2_MARGIN_THRESHOLD_STEP = 0.04

# ==================== EVALUATION CONFIGURATION ====================
# Macro F₀.₅ metric settings
F_BETA = 0.5  # F₀.₅ emphasizes precision 2× over recall
SINGLETON_THRESHOLD = 0.2  # Margin threshold for predicting empty (singleton)

# ==================== LOGGING & DEBUG ====================
VERBOSE = True
DEBUG_MODE = False
SAVE_INTERMEDIATE = True  # Save blocking results, features, etc.

# ==================== VALIDATION ====================
# Use val_sample for quick validation
USE_VAL_SAMPLE = True  # Set False to use full training data
VALIDATION_SIZE = 0.3  # 30% validation from training set

# ==================== PHASE MANAGEMENT ====================
def get_model_params(model_name: str, phase: int = PHASE) -> dict:
    """
    Get model parameters for specified phase.
    
    Args:
        model_name: 'xgboost', 'lightgbm', or 'catboost'
        phase: 1 or 2
    
    Returns:
        Dict of hyperparameters
    """
    if phase == 2:
        if model_name == "xgboost":
            return XGBOOST_PARAMS_PHASE2
        elif model_name == "lightgbm":
            return LIGHTGBM_PARAMS_PHASE2
        elif model_name == "catboost":
            return CATBOOST_PARAMS_PHASE2
    
    # Default Phase 1 params
    if model_name == "xgboost":
        return XGBOOST_PARAMS
    elif model_name == "lightgbm":
        return LIGHTGBM_PARAMS
    elif model_name == "catboost":
        return CATBOOST_PARAMS
    
    return {}


def get_threshold_ranges(phase: int = PHASE) -> Tuple[Tuple[float, float], float, Tuple[float, float], float]:
    """
    Get threshold optimization ranges for specified phase.
    
    Args:
        phase: 1 or 2
    
    Returns:
        (score_range, score_step, margin_range, margin_step)
    """
    if phase == 2:
        return (PHASE2_SCORE_THRESHOLD_RANGE, PHASE2_SCORE_THRESHOLD_STEP,
                PHASE2_MARGIN_THRESHOLD_RANGE, PHASE2_MARGIN_THRESHOLD_STEP)
    
    return (SCORE_THRESHOLD_RANGE, SCORE_THRESHOLD_STEP,
            MARGIN_THRESHOLD_RANGE, MARGIN_THRESHOLD_STEP)

print(f"✅ Config loaded from: {__file__}")
print(f"   PROJECT_ROOT: {PROJECT_ROOT}")
print(f"   DATA_DIR: {DATA_DIR}")
print(f"   PHASE: {PHASE}")
print(f"   NUM_FEATURES: {NUM_FEATURES} total (Phase 1: {NUM_FEATURES_PHASE1} + Phase 2 new: {NUM_FEATURES_PHASE2_NEW_TOTAL})")
print(f"     Phase 2 breakdown:")
print(f"       - Legal suffix: {NUM_FEATURES_LEGAL_SUFFIX}")
print(f"       - Composite: {NUM_FEATURES_COMPOSITE}")
print(f"       - Postal: {NUM_FEATURES_POSTAL}")
print(f"       - Landmark: {NUM_FEATURES_LANDMARK}")
print(f"       - Multi-channel (Task 4): {NUM_FEATURES_MULTICHANNEL}")
print(f"   ENSEMBLE_WEIGHTS: {ENSEMBLE_WEIGHTS}")
print(f"\n📋 Phase 2 Features Enabled:")
print(f"   - Legal suffix: {ENABLE_LEGAL_SUFFIX_FEATURES}")
print(f"   - Composite: {ENABLE_COMPOSITE_FEATURES}")
print(f"   - Postal hierarchical: {ENABLE_POSTAL_HIERARCHICAL}")
print(f"   - Landmark: {ENABLE_LANDMARK_FEATURES}")
print(f"   - Multi-channel: {ENABLE_MULTICHANNEL_FEATURES}")
print(f"   - Global consistency (Stage 7): {ENABLE_GLOBAL_CONSISTENCY}")
print(f"   - Embedding blocking (optional): {ENABLE_EMBEDDING_BLOCKING}")
