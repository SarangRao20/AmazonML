#!/usr/bin/env python3
"""
tools/rescore_calibrated.py

Solves the Density Shift / Over-Prediction Problem:
- Raw test inference used score_tau=0.65, resulting in mean matches = 4.31 (vs train ground truth 3.46).
- In F_0.5, each false positive carries a 4x penalty, dropping entity score from 1.0 to 0.714.
- This script scores only the ~7.46M candidate pairs in matching_results_dedup.tsv using the
  trained 3-way GBDT ensemble (XGBoost + LightGBM + CatBoost with 53 features).
- Applies optimal threshold tau >= 0.97 (which aligns mean matches to ~3.44 and empty singletons to ~5.0%).
- Runtime: ~15-18 minutes on 16 cores.
"""

import gc
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl
from code.business_entity_resolution.src.features import extract_features_from_records, RecView as _RecView
from code.business_entity_resolution.src.model import TriEnsembleModel

CHUNK_SIZE = 10_000
TAU = 0.975  # Optimal threshold matching train prior (mean 3.46, singletons 5.58%)


def main():
    t_start = time.time()
    print("=" * 60)
    print(f"Rescoring matching_results_dedup.tsv with Tri-Ensemble at tau={TAU}")
    print("=" * 60)

    # 1. Load models
    print("\n[1/4] Loading models...")
    t0 = time.time()
    model = TriEnsembleModel(verbose=False)
    models = model.load_models(ROOT / "models")
    print(f"  Models loaded in {time.time() - t0:.1f}s")

    # 2. Read matching_results_dedup.tsv
    print("\n[2/4] Reading matching_results_dedup.tsv...")
    t0 = time.time()
    dedup_path = ROOT / "output" / "matching_results_dedup.tsv"
    
    # Store s1_id -> list of candidate ids
    all_pairs = []  # list of (sid, [cands])
    total_pairs = 0
    with open(dedup_path, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            sid, _, rest = line.rstrip("\n").partition("\t")
            cands = [x.strip() for x in rest.split(",") if x.strip()]
            all_pairs.append((sid, cands))
            total_pairs += len(cands)

    print(f"  Read {len(all_pairs):,} S1 queries with {total_pairs:,} candidate pairs in {time.time() - t0:.1f}s")

    # 3. Load text records into memory
    print("\n[3/4] Loading text sources into memory dictionaries...")
    t0 = time.time()
    s1_df = pl.read_csv(ROOT / "dataset" / "test" / "test_source1.tsv", separator="\t")
    s1_records = dict(zip(s1_df["entity_id"], zip(s1_df["business_name"].fill_null(""), s1_df["business_address"].fill_null(""))))
    del s1_df
    gc.collect()

    s2_df = pl.read_csv(ROOT / "dataset" / "test" / "test_source2.tsv", separator="\t")
    s3_df = pl.read_csv(ROOT / "dataset" / "test" / "test_source3.tsv", separator="\t")
    s23_df = pl.concat([s2_df, s3_df])
    s23_records = dict(zip(s23_df["entity_id"], zip(s23_df["business_name"].fill_null(""), s23_df["business_address"].fill_null(""))))
    del s2_df, s3_df, s23_df
    gc.collect()
    print(f"  Loaded text dictionaries in {time.time() - t0:.1f}s (S1: {len(s1_records):,}, S2/S3: {len(s23_records):,})")

    # 4. Stream and score in chunks
    print(f"\n[4/4] Extracting features and scoring with tau >= {TAU}...")
    out_path = ROOT / "output" / "matching_results_calibrated_v2.tsv"
    rec_s1 = _RecView(s1_records)
    rec_s23 = _RecView(s23_records)

    total_kept = 0
    total_empty = 0
    n_chunks = (len(all_pairs) + CHUNK_SIZE - 1) // CHUNK_SIZE

    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        
        for chunk_idx in range(n_chunks):
            chunk_slice = all_pairs[chunk_idx * CHUNK_SIZE : (chunk_idx + 1) * CHUNK_SIZE]
            chunk_cands = {}
            for sid, cands in chunk_slice:
                if cands:
                    chunk_cands[sid] = [(c, 1.0) for c in cands]

            if not chunk_cands:
                for sid, _ in chunk_slice:
                    out_f.write(f"{sid}\t\n")
                    total_empty += 1
                continue

            # Feature extraction
            feats = extract_features_from_records(chunk_cands, rec_s1, rec_s23, phase=2, verbose=False)
            
            # Predict
            X = feats.drop(columns=["s1_id", "s2_s3_id"], errors="ignore")
            probs = model.predict_ensemble(X, models, use_calibration=True)

            # Group probabilities by sid
            by_sid = {}
            for sid, cid, p in zip(feats["s1_id"], feats["s2_s3_id"], probs):
                if p >= TAU:
                    by_sid.setdefault(sid, []).append((cid, float(p)))

            # Write results for this chunk
            for sid, _ in chunk_slice:
                cands_p = by_sid.get(sid, [])
                if not cands_p:
                    out_f.write(f"{sid}\t\n")
                    total_empty += 1
                else:
                    # Sort by probability descending
                    cands_p.sort(key=lambda t: (-t[1], t[0]))
                    kept_ids = [c for c, _ in cands_p][:11]
                    out_f.write(f"{sid}\t{','.join(kept_ids)}\n")
                    total_kept += len(kept_ids)

            if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
                elapsed = time.time() - t_start
                done_s1 = min(len(all_pairs), (chunk_idx + 1) * CHUNK_SIZE)
                rate = done_s1 / max(1, elapsed)
                print(f"  Chunk {chunk_idx + 1}/{n_chunks} ({done_s1:,}/{len(all_pairs):,} S1) | Kept: {total_kept:,} | Empty: {total_empty:,} ({100*total_empty/done_s1:.1f}%) | Mean: {total_kept/done_s1:.2f} | {elapsed:.0f}s elapsed ({rate:.0f} S1/s)")

    print("\n" + "=" * 60)
    print(f"Finished in {(time.time() - t_start) / 60:.1f} minutes!")
    print(f"Output saved to: {out_path}")
    print(f"Total S1: {len(all_pairs):,}")
    print(f"Total Kept Matches: {total_kept:,} (Mean: {total_kept/len(all_pairs):.3f})")
    print(f"Total Empty (Singletons): {total_empty:,} ({100*total_empty/len(all_pairs):.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
