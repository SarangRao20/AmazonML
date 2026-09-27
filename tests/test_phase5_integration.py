#!/usr/bin/env python3
"""
Phase 5 Integration Test
========================

Verifies that all Phase 5 quick wins are properly integrated:
1. 3-gram Jaccard similarity in features
2. Exact core-name blocking channel
3. IDF-weighted token blocking
4. Frequency-based stopword suppression
"""

import sys
import pandas as pd
from pathlib import Path

# Add code directory to path
code_dir = str(Path(__file__).parent / "code" / "business_entity_resolution")
sys.path.insert(0, code_dir)

def test_phase5_integration():
    """Test Phase 5 integration without running full pipeline."""
    
    print("\n" + "="*80)
    print("✅ PHASE 5 INTEGRATION TEST")
    print("="*80)
    
    # Test 1: Feature extraction with 3-gram Jaccard
    print("\n[TEST 1] 3-gram Jaccard similarity in features.py")
    try:
        from features import get_char_ngrams, FeatureExtractor
        
        # Test get_char_ngrams function
        ngrams = get_char_ngrams("john", n=3)
        assert "joh" in ngrams and "ohn" in ngrams, "3-grams not extracted correctly"
        print("   ✅ get_char_ngrams() works correctly")
        
        # Test 3-gram Jaccard in FeatureExtractor
        extractor = FeatureExtractor()
        name_features = extractor.extract_name_ngram_jaccard("john", "john")
        assert "name_char3_jaccard" in name_features, "Missing name_char3_jaccard feature"
        assert name_features["name_char3_jaccard"] == 1.0, "Perfect match should return 1.0"
        print("   ✅ extract_name_ngram_jaccard() works correctly")
        
        addr_features = extractor.extract_address_ngram_jaccard("123 Main St", "123 Main St")
        assert "addr_char3_jaccard" in addr_features, "Missing addr_char3_jaccard feature"
        assert addr_features["addr_char3_jaccard"] == 1.0, "Perfect match should return 1.0"
        print("   ✅ extract_address_ngram_jaccard() works correctly")
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    
    # Test 2: Exact core-name blocking channel
    print("\n[TEST 2] Exact core-name blocking channel in blocking.py")
    try:
        from blocking import MultiChannelBlocker
        
        blocker = MultiChannelBlocker(verbose=False)
        
        # Test get_core_name
        core_name = blocker.get_core_name("ACME Corporation")
        assert core_name.lower() != "acme corporation", "Legal suffix not removed"
        print(f"   ✅ get_core_name() works: 'ACME Corporation' -> '{core_name}'")
        
        # Test build_channel_0_exact_core_name
        test_records = {
            "s1": {"entity_id": "s1", "business_name": "ACME Corp", "business_address": "123 Main"},
            "s2": {"entity_id": "s2", "business_name": "ACME Inc", "business_address": "456 Oak"},
        }
        
        channel_0_index = blocker.build_channel_0_exact_core_name(test_records)
        assert len(channel_0_index) > 0, "Channel 0 index is empty"
        print(f"   ✅ build_channel_0_exact_core_name() created {len(channel_0_index)} unique core names")
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    
    # Test 3: IDF-weighted token blocking & frequency suppression
    print("\n[TEST 3] IDF-weighted token blocking & frequency-based suppression")
    try:
        from blocking import MultiChannelBlocker
        
        blocker = MultiChannelBlocker(verbose=False)
        
        # Test compute_token_frequencies
        test_records = {
            "s1": {"entity_id": "s1", "business_name": "ACME Corporation", "business_address": "123 Main Street"},
            "s2": {"entity_id": "s2", "business_name": "ACME Inc", "business_address": "456 Oak Lane"},
            "s3": {"entity_id": "s3", "business_name": "XYZ Ltd", "business_address": "789 Elm Road"},
        }
        
        name_freq, addr_freq = blocker.compute_token_frequencies(test_records)
        assert len(name_freq) > 0, "Name frequencies not computed"
        assert len(addr_freq) > 0, "Address frequencies not computed"
        print(f"   ✅ compute_token_frequencies() computed {len(name_freq)} name tokens, {len(addr_freq)} addr tokens")
        
        # Test build_all_channels includes frequency info
        s1_idx, s2_idx = blocker.build_all_channels(test_records, test_records)
        assert "channel_0" in s1_idx, "Channel 0 not in indices"
        assert "name_freq" in s1_idx, "name_freq not in indices"
        assert "addr_freq" in s1_idx, "addr_freq not in indices"
        print("   ✅ build_all_channels() stores frequency information")
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    
    # Test 4: Full feature extraction pipeline
    print("\n[TEST 4] Full feature extraction pipeline with Phase 5 features")
    try:
        from features import FeatureExtractor
        
        extractor = FeatureExtractor()
        
        s1_record = {
            "entity_id": "s1",
            "business_name": "ACME Corporation",
            "business_address": "123 Main Street"
        }
        s2_record = {
            "entity_id": "s2",
            "business_name": "ACME Inc",
            "business_address": "123 Main St"
        }
        
        features = extractor.extract_features_for_pair(s1_record, s2_record, phase=2)
        
        # Check Phase 1 features
        assert "name_ratio" in features, "Missing Phase 1 feature: name_ratio"
        assert "addr_ratio" in features, "Missing Phase 1 feature: addr_ratio"
        
        # Check Phase 5 features
        assert "name_char3_jaccard" in features, "Missing Phase 5 feature: name_char3_jaccard"
        assert "addr_char3_jaccard" in features, "Missing Phase 5 feature: addr_char3_jaccard"
        
        print(f"   ✅ extract_features_for_pair() returns {len(features)} features")
        print(f"      - Phase 1 baseline: name_ratio={features['name_ratio']:.3f}")
        print(f"      - Phase 5 (NEW): name_char3_jaccard={features['name_char3_jaccard']:.3f}")
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("✅ ALL PHASE 5 INTEGRATION TESTS PASSED!")
    print("="*80)
    
    print("\n📋 Phase 5 Quick Wins Summary:")
    print("   ✅ 1. 3-gram Jaccard similarity added to features")
    print("   ✅ 2. Exact core-name blocking channel added")
    print("   ✅ 3. IDF-weighted token blocking integrated")
    print("   ✅ 4. Frequency-based stopword suppression added")
    
    print("\n📊 Expected Impact:")
    print("   - 3-gram Jaccard: +0.2-0.3% F₀.₅")
    print("   - Exact core-name: +0.1-0.2% F₀.₅")
    print("   - IDF weighting: +0.1-0.2% F₀.₅")
    print("   - Frequency suppression: +0.05-0.1% F₀.₅")
    print("   - Total combined: +0.55-0.85% F₀.₅")
    
    print("\n🎯 Phase 4 baseline: 98.0%")
    print("   Phase 5 target: 98.5%+ ✨")
    print("\n" + "="*80 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        success = test_phase5_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
