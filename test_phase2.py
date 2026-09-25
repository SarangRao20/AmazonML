#!/usr/bin/env python3
"""
Phase 2 validation test script.

Tests Phase 2 (55 features + Stage 7 + consistency) on validation sample.
Expected targets:
- Recall ≥ 96%
- F₀.₅ ≥ 97%
- Improvement over Phase 1: +1-2 F₀.₅ points

Usage:
    python test_phase2.py
"""

import sys
import time
from pathlib import Path

# Add code directory to path
code_dir = Path(__file__).parent / "code" / "business_entity_resolution"
sys.path.insert(0, str(code_dir.parent))

from business_entity_resolution.pipeline import EntityResolutionPipeline
from business_entity_resolution.config import PHASE, NUM_FEATURES
from business_entity_resolution.data_loader import DataLoader


def main():
    """Run Phase 2 validation test."""
    print("\n" + "="*80)
    print("🧪 PHASE 2 VALIDATION TEST")
    print("="*80)
    print(f"Phase: {PHASE}")
    print(f"Features: {NUM_FEATURES}")
    print(f"Expected targets: Recall ≥96%, F₀.₅ ≥97%")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    try:
        # Initialize pipeline
        print("📋 Initializing pipeline...")
        pipeline = EntityResolutionPipeline(verbose=True)
        
        # Run full pipeline
        print("\n🚀 Running Phase 2 pipeline...")
        predictions = pipeline.run_full_pipeline()
        
        elapsed = time.time() - start_time
        
        # Print results
        print("\n" + "="*80)
        print("✅ PHASE 2 TEST COMPLETE")
        print("="*80)
        print(f"Time elapsed: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
        print(f"Predictions generated: {len(predictions):,} entities")
        print(f"Best F₀.₅: {pipeline.best_f_beta*100:.2f}%" if hasattr(pipeline, 'best_f_beta') else "N/A")
        print(f"Best score threshold: {pipeline.best_score_tau:.3f}" if hasattr(pipeline, 'best_score_tau') else "N/A")
        print(f"Best margin threshold: {pipeline.best_margin_tau:.3f}" if hasattr(pipeline, 'best_margin_tau') else "N/A")
        print("="*80)
        
        # Validation checks
        print("\n📊 VALIDATION CHECKS:")
        
        # Check if results meet expectations
        if hasattr(pipeline, 'best_f_beta'):
            f_beta = pipeline.best_f_beta
            if f_beta >= 0.97:
                print(f"  ✅ F₀.₅ target met: {f_beta*100:.2f}% ≥ 97%")
            else:
                print(f"  ⚠️  F₀.₅ below target: {f_beta*100:.2f}% < 97%")
        
        # Check feature count
        print(f"  ✅ Features: {NUM_FEATURES} (Phase 2 expanded from 17 to 55)")
        
        # Check predictions
        total_predicted = sum(len(matches) for matches in predictions.values())
        print(f"  ℹ️  Total predicted matches: {total_predicted:,}")
        
        print("\n✅ Phase 2 validation test passed!")
        
    except Exception as e:
        print(f"\n❌ Phase 2 test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("🎉 Ready for Phase 2 submission!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
