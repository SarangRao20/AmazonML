#!/usr/bin/env python3
"""
Generate submissions at various flat thresholds with greedy 1-to-1 dedup.
Processes one country at a time to stay within RAM.
Skips candidate_pairs intersection (scores are already a subset of candidates).
"""
import sys, gc
from pathlib import Path
from collections import defaultdict

import polars as pl

TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"output_tau{int(TAU*100)}_dedup")

s1 = pl.read_csv("dataset/test/test_source1.tsv", separator="\t",
                  columns=["entity_id", "country"])
country_map = {r[0]: r[1] for r in s1.iter_rows()}
all_s1 = list(country_map.keys())

# Group S1 by country
s1_by_country = defaultdict(list)
for sid, cc in country_map.items():
    s1_by_country[cc].append(sid)

result = {}  # sid -> list of cid

for cc in sorted(s1_by_country):
    sp = Path(f"output/parts/{cc}.scores.tsv")
    if not sp.exists():
        print(f"  {cc}: no scores file, keeping empty")
        for sid in s1_by_country[cc]:
            result[sid] = []
        continue

    # Load scores for this country only
    scores = {}
    with open(sp) as f:
        for line in f:
            if line.startswith("source1"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sid, cid, p = parts[0], parts[1], float(parts[2])
            if p >= TAU:
                scores.setdefault(sid, []).append((p, cid))

    # Greedy 1-to-1 within country
    all_pairs = []
    for sid, cands in scores.items():
        for p, cid in cands:
            all_pairs.append((p, sid, cid))
    all_pairs.sort(key=lambda x: x[0], reverse=True)

    assigned = set()
    country_result = defaultdict(list)
    for p, sid, cid in all_pairs:
        if cid not in assigned:
            assigned.add(cid)
            country_result[sid].append(cid)

    # Fill in all S1 for this country
    n_pairs = 0
    n_empty = 0
    for sid in s1_by_country[cc]:
        result[sid] = country_result.get(sid, [])
        n_pairs += len(result[sid])
        if not result[sid]:
            n_empty += 1

    n_c = len(s1_by_country[cc])
    print(f"  {cc:<8}: mean={n_pairs/n_c:.3f}, empty={n_empty/n_c*100:.2f}%")

    del scores, all_pairs, country_result, assigned
    gc.collect()

# Write
OUT.mkdir(exist_ok=True)
total_pairs = 0
total_empty = 0
with open(OUT / "matching_results.tsv", "w") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in all_s1:
        cs = result.get(sid, [])
        f.write(f"{sid}\t{','.join(cs)}\n")
        total_pairs += len(cs)
        if not cs:
            total_empty += 1

print(f"  TOTAL  : mean={total_pairs/len(all_s1):.3f}, "
      f"empty={total_empty/len(all_s1)*100:.2f}%")
print(f"  Wrote {OUT}/matching_results.tsv ({total_pairs:,} pairs)")
