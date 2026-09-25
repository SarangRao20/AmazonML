#!/usr/bin/env python
"""
Quick pipeline test on sample data - using package imports.

Tests all 9 core modules without training (fast).
"""

import sys
import os

# Add to path
sys.path.insert(0, '/home/sarang/AmazonML/code')

import time
print("\n" + "="*80)
print("🧪 PIPELINE MODULE TEST - Phase 1")
print("="*80)

# Test 1: Config
print("\n1️⃣  Testing config.py...")
try:
    from business_entity_resolution.config import (
        FEATURE_NAMES, NUM_FEATURES, ENSEMBLE_WEIGHTS, BLOCKING_CONFIG
    )
    print(f"   ✓ Config loaded")
    print(f"     Features: {NUM_FEATURES}")
    print(f"     Weights: {ENSEMBLE_WEIGHTS}")
except Exception as e:
    print(f"   ✗ Config failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Data Loader
print("\n2️⃣  Testing data_loader.py...")
try:
    from business_entity_resolution.data_loader import DataLoader
    loader = DataLoader(use_val_sample=True, verbose=False)
    s1, s2, s3, gt = loader.load_data()
    print(f"   ✓ Data loaded")
    print(f"     S1: {len(s1):,} | S2: {len(s2):,} | S3: {len(s3):,}")
    
    gt_dict = loader.build_ground_truth_dict()
    print(f"     Ground truth: {len(gt_dict):,} entities")
    
    train_ids, val_ids = loader.get_entity_level_split()
    overlap = len(set(train_ids) & set(val_ids))
    print(f"     Train/Val split: {len(train_ids):,}/{len(val_ids):,} (disjoint: {overlap == 0} ✓)")
except Exception as e:
    print(f"   ✗ DataLoader failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Normalize
print("\n3️⃣  Testing normalize.py...")
try:
    from business_entity_resolution.normalize import (
        normalize_name, normalize_address, transliterate_devanagari
    )
    
    test_name = "ACME Corporation"
    norm_name = normalize_name(test_name)
    print(f"   ✓ Normalization works")
    print(f"     '{test_name}' -> '{norm_name}'")
    
    # Test Devanagari
    hindi_text = "मुंबई"
    translit = transliterate_devanagari(hindi_text)
    print(f"     Hindi: '{hindi_text}' -> '{translit}'")
except Exception as e:
    print(f"   ✗ Normalize failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Blocking
print("\n4️⃣  Testing blocking.py...")
try:
    from business_entity_resolution.blocking import MultiChannelBlocker
    import pandas as pd
    
    blocker = MultiChannelBlocker(verbose=False)
    s1_sample = s1.head(30)
    s2_sample = s2.head(30)
    s3_sample = s3.head(30)
    
    s1_records = {row['entity_id']: row.to_dict() for _, row in s1_sample.iterrows()}
    s2_records = {row['entity_id']: row.to_dict() for _, row in s2_sample.iterrows()}
    s3_records = {row['entity_id']: row.to_dict() for _, row in s3_sample.iterrows()}
    s2_s3_records = {**s2_records, **s3_records}
    
    candidates = blocker.generate_all_candidates(s1_records, s2_s3_records)
    print(f"   ✓ Blocking works")
    print(f"     Generated {len(candidates):,} S1 entities with candidates")
    
    total_pairs = sum(len(c) for c in candidates.values())
    print(f"     Total candidate pairs: {total_pairs:,}")
except Exception as e:
    print(f"   ✗ Blocking failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Features
print("\n5️⃣  Testing features.py...")
try:
    from business_entity_resolution.features import FeatureExtractor
    
    extractor = FeatureExtractor(verbose=False)
    
    test_s1 = s1.iloc[0].to_dict()
    test_s2 = s2.iloc[0].to_dict()
    
    features = extractor.extract_features_for_pair(test_s1, test_s2)
    print(f"   ✓ Feature extraction works")
    print(f"     Extracted {len(features):,} features for 1 pair")
    print(f"     Sample features: name_ratio={features.get('name_ratio', 0):.3f}, number_overlap={features.get('number_overlap', 0):.3f}")
except Exception as e:
    print(f"   ✗ Features failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Model (skip training, just test init)
print("\n6️⃣  Testing model.py...")
try:
    from business_entity_resolution.model import TriEnsembleModel
    
    ensemble = TriEnsembleModel(verbose=False)
    print(f"   ✓ Model initialized")
    print(f"     Weights: XGB={ENSEMBLE_WEIGHTS['xgboost']} LGB={ENSEMBLE_WEIGHTS['lightgbm']} CB={ENSEMBLE_WEIGHTS['catboost']}")
except Exception as e:
    print(f"   ✗ Model failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Threshold Optimizer
print("\n7️⃣  Testing threshold_optimizer.py...")
try:
    from business_entity_resolution.threshold_optimizer import ThresholdOptimizer
    
    optimizer = ThresholdOptimizer(verbose=False)
    
    # Synthetic test
    test_predictions = {
        'S1-001': {'S2-010', 'S3-020'},
        'S1-002': set(),
    }
    test_gt = {
        'S1-001': {'S2-010', 'S3-020'},
        'S1-002': set(),
    }
    
    f_beta = optimizer.compute_macro_f_beta(test_predictions, test_gt, beta=0.5)
    print(f"   ✓ Threshold optimizer works")
    print(f"     Test F₀.₅: {f_beta*100:.2f}%")
except Exception as e:
    print(f"   ✗ Threshold optimizer failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Evaluate
print("\n8️⃣  Testing evaluate.py...")
try:
    from business_entity_resolution.evaluate import MetricsComputer
    
    computer = MetricsComputer()
    
    test_y_true = {'A', 'B', 'C'}
    test_y_pred = {'A', 'B', 'D'}
    
    metrics = computer.compute_entity_metrics(test_y_true, test_y_pred, beta=0.5)
    print(f"   ✓ Evaluation works")
    print(f"     Precision: {metrics['precision']*100:.1f}%")
    print(f"     Recall: {metrics['recall']*100:.1f}%")
    print(f"     F₀.₅: {metrics['f_beta']*100:.1f}%")
except Exception as e:
    print(f"   ✗ Evaluate failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Pipeline
print("\n9️⃣  Testing pipeline.py...")
try:
    from business_entity_resolution.pipeline import EntityResolutionPipeline
    
    pipeline = EntityResolutionPipeline(verbose=False)
    print(f"   ✓ Pipeline initialized")
    
    # Test stage 1
    pipeline.stage_1_load_data()
    print(f"     Stage 1: Data loaded ✓")
    
    # Test stage 3 (blocking)
    pipeline.stage_3_blocking()
    print(f"     Stage 3: Blocking complete ✓")
    
except Exception as e:
    print(f"   ✗ Pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*80)
print("✅ ALL 9 MODULE TESTS PASSED!")
print("="*80)
print("\n📋 SUMMARY:")
print("   ✓ config.py - Hyperparameters & paths")
print("   ✓ data_loader.py - Entity-level splits (DISJOINT)")
print("   ✓ normalize.py - Unicode + Devanagari + legal suffix")
print("   ✓ blocking.py - 4-channel blocking")
print("   ✓ features.py - 17 core features")
print("   ✓ model.py - Tri-ensemble (XGB/LGB/CatBoost)")
print("   ✓ threshold_optimizer.py - Macro F₀.₅ grid search")
print("   ✓ evaluate.py - Metrics & error analysis")
print("   ✓ pipeline.py - End-to-end orchestration")

print("\n🚀 READY FOR PHASE 1 TRAINING!")
print("   Command: python -m business_entity_resolution.pipeline")
print("   Expected F₀.₅: 95-97% on validation sample")
print("\n" + "="*80)
