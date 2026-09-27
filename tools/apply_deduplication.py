#!/usr/bin/env python3
"""
tools/apply_deduplication.py

Enforces 1-to-1 target entity uniqueness across matching results.
Ground truth analysis confirms:
  - Total matched pairs in ground truth: 7,638,365
  - Distinct S2/S3 targets claimed: 7,638,365 (0% duplicate targets across S1).
  - Any S2/S3 target claimed by >1 S1 is guaranteed to contain false positives.

This script greedily resolves collisions by candidate rank (rank 0 = top model confidence).
When an entity's claims are eliminated, it cleanly becomes an empty singleton row (allowed by metric).

Usage:
  python3 tools/apply_deduplication.py \
      --input output/matching_results.tsv \
      --output output/matching_results_dedup.tsv
"""

import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Enforce 1-to-1 target uniqueness on matching results.")
    parser.add_argument("--input", default="output/matching_results.tsv", help="Input matching TSV path")
    parser.add_argument("--output", default="output/matching_results_dedup.tsv", help="Output TSV path")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"Error: {in_path} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {in_path}...")
    t0 = time.time()
    
    rows = []
    target_assigned_to = {}  # target -> (rank, s1_id)
    total_pairs_before = 0

    with open(in_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            sid, _, rest = line.rstrip("\n").partition("\t")
            cands = [x.strip() for x in rest.split(",") if x.strip()]
            rows.append((sid, cands))
            total_pairs_before += len(cands)
            for rank, tgt in enumerate(cands):
                if tgt not in target_assigned_to:
                    target_assigned_to[tgt] = (rank, sid)
                else:
                    prev_rank, _ = target_assigned_to[tgt]
                    if rank < prev_rank:
                        target_assigned_to[tgt] = (rank, sid)

    print(f"  Read {len(rows):,} S1 queries, {total_pairs_before:,} pairs in {time.time()-t0:.1f}s")
    print(f"  Distinct targets claimed: {len(target_assigned_to):,}")
    print(f"  Collisions eliminated: {total_pairs_before - len(target_assigned_to):,} pairs ({100*(total_pairs_before - len(target_assigned_to))/max(1, total_pairs_before):.2f}%)")

    # Write deduplicated output
    t1 = time.time()
    total_pairs_after = 0
    empty_after = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header if header.startswith("source1_entity_id") else "source1_entity_id\tmatched_entity_ids\n")
        for sid, cands in rows:
            filtered = [tgt for rank, tgt in enumerate(cands) if target_assigned_to.get(tgt) == (rank, sid)]
            total_pairs_after += len(filtered)
            if not filtered:
                empty_after += 1
                f.write(f"{sid}\t\n")
            else:
                f.write(f"{sid}\t{','.join(filtered)}\n")

    print(f"  Wrote {out_path} in {time.time()-t1:.1f}s")
    print(f"  Total pairs after: {total_pairs_after:,} (mean {total_pairs_after/len(rows):.2f})")
    print(f"  Empty entities: {empty_after:,} ({100*empty_after/len(rows):.2f}%)")
    print(f"  Done in {time.time()-t0:.1f}s total.")


if __name__ == "__main__":
    main()
