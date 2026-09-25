"""Functional smoke test: measure real blocking recall + feature extraction.

This is the first honest measurement in the repo. It answers the only
question that matters for F_0.5: what fraction of true matches does the
blocker actually retrieve?

Run:
    python smoke_test.py [--limit 5000]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from code.business_entity_resolution.src.config import FEATURE_NAMES, NUM_FEATURES
from code.business_entity_resolution.src.data_loader import DataLoader
from code.business_entity_resolution.src.features import FeatureExtractor
from code.business_entity_resolution.src.evaluate import MetricsComputer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5000,
                    help="Number of S1 entities to block for")
    args = ap.parse_args()

    print("=" * 70)
    print("SMOKE TEST: blocking recall + feature extraction")
    print("=" * 70)

    print("\n[1/4] Loading val_sample ...")
    t0 = time.time()
    loader = DataLoader(use_val_sample=True, verbose=True)
    s1, s2, s3, gt = loader.load_data()
    print(f"      loaded in {time.time() - t0:.1f}s")

    gt_dict = loader.build_ground_truth_dict()
    s1_dict = {r["entity_id"]: r for r in s1.to_dict("records")}
    s2_s3_dict = {r["entity_id"]: r for r in
                  __import__("pandas").concat([s2, s3]).to_dict("records")}

    n_singletons = sum(1 for v in gt_dict.values() if not v)
    print(f"\n      S1 entities:        {len(s1_dict):,}")
    print(f"      S2+S3 pool:         {len(s2_s3_dict):,}")
    print(f"      ground truth rows:  {len(gt_dict):,}")
    print(f"      singletons:         {n_singletons:,} "
          f"({100 * n_singletons / max(len(gt_dict), 1):.2f}%)")

    # Restrict to a sample of S1 for speed
    target_ids = list(gt_dict.keys())[: args.limit]
    s1_sub = {k: s1_dict[k] for k in target_ids if k in s1_dict}
    gt_sub = {k: gt_dict[k] for k in s1_sub}

    print(f"\n[2/4] Blocking {len(s1_sub):,} S1 entities against "
          f"{len(s2_s3_dict):,} pool ...")
    t0 = time.time()
    from code.business_entity_resolution.src.blocking import MultiChannelBlocker
    blocker = MultiChannelBlocker(verbose=True)
    candidates = blocker.generate_all_candidates(s1_sub, s2_s3_dict)
    print(f"      candidates generated in {time.time() - t0:.1f}s")

    print("\n[3/4] BLOCKING RECALL (the critical metric)")
    recall = blocker.measure_recall(candidates, gt_sub)
    n_cand = [len(v) for v in candidates.values()]
    if n_cand:
        import statistics
        print(f"\n      candidates/entity: mean {statistics.mean(n_cand):.1f} "
              f"median {statistics.median(n_cand):.0f} "
              f"max {max(n_cand)}")

    # full-coverage: entities where ALL true matches were retrieved
    full = 0
    non_single = 0
    for s1_id, truth in gt_sub.items():
        if not truth:
            continue
        non_single += 1
        got = {c for c, _ in candidates.get(s1_id, [])}
        if truth.issubset(got):
            full += 1
    if non_single:
        print(f"      entity FULL coverage: {full}/{non_single} "
              f"({100 * full / non_single:.2f}%)")

    print(f"\n[4/4] Feature extraction check")
    ex = FeatureExtractor()
    sample_pair = next(
        ((s1_id, c) for s1_id, v in candidates.items() for c, _ in v
         if gt_sub.get(s1_id) and c in gt_sub[s1_id]),
        None,
    )
    if sample_pair:
        s1_id, cand_id = sample_pair
        feats = ex.extract_features_for_pair(s1_sub[s1_id], s2_s3_dict[cand_id])
        print(f"      true pair ({s1_id}, {cand_id}) -> {len(feats)} features")
        print(f"      config expects NUM_FEATURES={NUM_FEATURES}, "
              f"len(FEATURE_NAMES)={len(FEATURE_NAMES)}")
        if len(feats) != NUM_FEATURES:
            print(f"      MISMATCH: extractor returned {len(feats)}, "
                  f"config declares {NUM_FEATURES}")
        missing = set(FEATURE_NAMES) - set(feats)
        extra = set(feats) - set(FEATURE_NAMES)
        if missing:
            print(f"      missing from extractor: {sorted(missing)[:8]}")
        if extra:
            print(f"      not declared in config: {sorted(extra)[:8]}")
    else:
        print("      no true pair available in sample to check")

    # metric sanity
    print("\n[metric] verifying macro F_0.5 against the PS worked example")
    # PS: pred {a,b,c}, truth {a,c} -> P=2/3 R=1.0 F=0.714
    m = MetricsComputer.compute_entity_metrics({"a", "c"}, {"a", "b", "c"}, beta=0.5)
    print(f"      pred={{a,b,c}} truth={{a,c}} -> P={m['precision']:.3f} "
          f"R={m['recall']:.3f} F0.5={m['f_beta']:.3f}  (PS says 0.714)")
    # singleton: truth empty, pred empty -> 1.0
    ms = MetricsComputer.compute_entity_metrics(set(), set(), beta=0.5)
    print(f"      singleton truth={{}} pred={{}} -> F0.5={ms['f_beta']:.3f}  "
          f"(PS says 1.0)")

    print("\n" + "=" * 70)
    print(f"BLOCKING RECALL: {recall * 100:.2f}%")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
