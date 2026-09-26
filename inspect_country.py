"""Audit one country's finished part file. Runs alongside inference.

France is the country that most needs this: it never appears in train, so
its score floor is a heuristic (FR_DELTA) rather than something the model
learned. Cheap checks that catch a bad France early:

  * every S1 entity of that country present exactly once
  * no ID repeated inside a match list, no S1- self-match
  * every matched ID exists in that country's own pool
  * no cross-country leakage (a France S1 matching a US-pool record)
  * match-count distribution, and how much of it is the MAX_PER_S1 cap
"""
import sys, collections
import polars as pl

country = sys.argv[1] if len(sys.argv) > 1 else "France"
part = f"output/parts/{country}"
MAX_PER_S1 = 11

s1 = (pl.scan_csv("dataset/test/test_source1.tsv", separator="\t",
                  infer_schema_length=0)
        .filter(pl.col("country") == country)
        .select("entity_id").collect().to_series().to_list())
s1_set = set(s1)
print(f"{country}: {len(s1):,} S1 entities in test, {len(s1_set):,} unique")

pool_country = {}
for f in ("test_source2", "test_source3"):
    d = (pl.scan_csv(f"dataset/test/{f}.tsv", separator="\t", infer_schema_length=0)
           .select("entity_id", "country").collect())
    for e, c in zip(d["entity_id"], d["country"]):
        pool_country[e] = c
print(f"pool ids: {len(pool_country):,}")

rows, dup_rows, empty, intra_dupe, self_match, unknown, leak = 0, 0, 0, 0, 0, 0, 0
per_entity, bad_examples, at_cap = [], [], 0
seen = set()
leak_ex, unknown_ex = [], []

with open(f"{part}.matching.tsv", encoding="utf-8") as fh:
    header = next(fh).rstrip("\n").split("\t")
    for line in fh:
        rows += 1
        sid, _, rest = line.rstrip("\n").partition("\t")
        if sid in seen:
            dup_rows += 1
        seen.add(sid)
        ids = [x for x in rest.split(",") if x]
        if not ids:
            empty += 1
            per_entity.append(0)
            continue
        if len(ids) != len(set(ids)):
            intra_dupe += 1
        per_entity.append(len(ids))
        if len(ids) >= MAX_PER_S1:
            at_cap += 1
        for m in ids:
            if m.startswith("S1-"):
                self_match += 1
            elif m not in pool_country:
                unknown += 1
                if len(unknown_ex) < 3: unknown_ex.append((sid, m))
            elif pool_country[m] != country:
                leak += 1
                if len(leak_ex) < 3: leak_ex.append((sid, m, pool_country[m]))
        if len(per_entity) <= 3 and ids:
            bad_examples.append((sid, len(ids)))

print(f"\nrows={rows:,}  unique S1 rows={len(seen):,}  expected={len(s1_set):,}")
print(f"missing S1 = {len(s1_set - seen):,}   extra S1 = {len(seen - s1_set):,}")
print(f"duplicate S1 rows = {dup_rows}")
print(f"header = {header}")

pe = sorted(per_entity)
n = len(pe)
def pct(p): return pe[min(n - 1, int(n * p))]
print(f"\nempty rows            : {empty:,} ({100*empty/n:.2f}%)")
print(f"matches per non-empty : mean {sum(pe)/n:.2f}  median {pct(0.5)}  "
      f"p90 {pct(0.9)}  p99 {pct(0.99)}  max {pe[-1]}")
hist = collections.Counter(pe)
print("distribution          :", {k: hist[k] for k in sorted(hist)[:12]})
print(f"at MAX_PER_S1 cap ({MAX_PER_S1}) : {at_cap:,} ({100*at_cap/n:.2f}%)")

print("\nintegrity:")
for label, val, ex in (("repeated ID inside a list", intra_dupe, None),
                       ("S1- self match", self_match, None),
                       ("matched ID not in pool", unknown, unknown_ex),
                       ("cross-country leak", leak, leak_ex)):
    flag = "OK" if val == 0 else "PROBLEM"
    print(f"  {flag:8} {label}: {val}" + (f"   e.g. {ex}" if ex else ""))
