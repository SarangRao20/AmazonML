"""Memory-efficient, country-partitioned streaming inference pipeline for Amazon ML Challenge 2026.

Strategy:
- Partition strictly by country (verified: 0 cross-country matches).
- Block with the MultiPassBlocker but keep candidates as integer pool positions
  (no string materialization) to save memory.
- Build attribute lookups LAZILY: only for Source-1 records and for the pool
  rows actually referenced by blocking candidates. On India (4.7M-row pool)
  this is several GB cheaper than precomputing attributes for every pool record.
- Score in batches with LightGBM, write results per country, then free memory.
"""

import os
import gc
import argparse
import time
from functools import lru_cache
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from tqdm import tqdm

from .preprocess import (
    normalize_text,
    get_core_name,
    get_tokens,
    get_char_ngrams,
    get_numbers
)
from .blocking import MultiPassBlocker
from .features import extract_pair_features

ATTR_COLS = ["entity_id", "business_name", "business_address"]


def load_country_partition(path: str, country: str) -> pd.DataFrame:
    """Return only entity_id / business_name / business_address for one country."""
    filtered = (
        pl.read_csv(path, separator="\t", infer_schema_length=0)
        .filter(pl.col("country") == country)
        .select(ATTR_COLS)
    )
    if filtered.height == 0:
        return pd.DataFrame(columns=ATTR_COLS)
    return pd.DataFrame({column: filtered[column].to_list() for column in ATTR_COLS}).fillna("")


def block_country_positions(blocker, s1_df: pd.DataFrame, pool_df: pd.DataFrame):
    """Block one country, returning flat candidate arrays and row offsets."""
    idx_data = blocker.build_index(pool_df)
    pool_ids = idx_data["pool_ids"].copy()

    s1_ids = s1_df["entity_id"].values
    s1_names = s1_df["business_name"].values
    s1_addrs = s1_df["business_address"].values

    n_s1 = len(s1_df)
    candidate_pool_pos = np.empty(n_s1 * blocker.top_k, dtype=np.int32)
    candidate_scores = np.empty(n_s1 * blocker.top_k, dtype=np.float32)
    offsets = np.zeros(n_s1 + 1, dtype=np.int64)
    write_pos = 0

    for i in range(n_s1):
        norm_name = normalize_text(s1_names[i])
        core_name = get_core_name(norm_name)
        norm_addr = normalize_text(s1_addrs[i])

        n_toks = get_tokens(core_name, min_len=3)
        a_toks = get_tokens(norm_addr, min_len=3)
        nums = get_numbers(norm_addr)

        cand_items = blocker.block_entity(core_name, n_toks, a_toks, nums, idx_data)
        end_pos = write_pos + len(cand_items)
        if cand_items:
            candidate_pool_pos[write_pos:end_pos] = [c_idx for c_idx, _ in cand_items]
            candidate_scores[write_pos:end_pos] = [score for _, score in cand_items]
            write_pos = end_pos
        offsets[i + 1] = write_pos

    del idx_data
    gc.collect()
    return (
        candidate_pool_pos[:write_pos],
        candidate_scores[:write_pos],
        offsets,
        s1_ids,
        pool_ids,
    )


def compute_attrs(name: str, addr: str):
    """Compute the 7-element attribute tuple used by extract_pair_features."""
    norm_name = normalize_text(name)
    core_name = get_core_name(norm_name)
    norm_addr = normalize_text(addr)
    return (
        norm_name,
        core_name,
        norm_addr,
        get_tokens(core_name, 3),
        get_tokens(norm_addr, 3),
        get_numbers(norm_addr),
        get_char_ngrams(core_name, 3)
    )


def run_country_inference(
    country: str,
    s1_path: str,
    s2_path: str,
    s3_path: str,
    model: lgb.Booster,
    threshold: float,
    top_k: int,
    blocker: MultiPassBlocker
):
    print(f"\n{'='*20} Processing Country: {country} {'='*20}")
    t0 = time.time()

    # 1. Load country slices (only the 3 needed columns)
    print(f"Loading {country} partitions...")
    c_s1 = load_country_partition(s1_path, country)
    c_s2 = load_country_partition(s2_path, country)
    c_s3 = load_country_partition(s3_path, country)
    c_pool = pd.concat([c_s2, c_s3], ignore_index=True)
    del c_s2, c_s3
    gc.collect()

    n_s1 = len(c_s1)
    n_pool = len(c_pool)
    print(f"Loaded {n_s1:,} S1 entities against {n_pool:,} pool records in {time.time()-t0:.2f}s.")

    if n_s1 == 0 or n_pool == 0:
        return [], []

    # 2. Country-specific blocking (position keyed, index freed after)
    t_b = time.time()
    candidate_pool_pos, candidate_scores, offsets, s1_ids, pool_ids = block_country_positions(
        blocker, c_s1, c_pool
    )
    total_pairs = len(candidate_pool_pos)
    print(f"Blocking complete in {time.time()-t_b:.2f}s. Total candidate pairs: {total_pairs:,}")

    s1_names = c_s1["business_name"].to_numpy(copy=True)
    s1_addrs = c_s1["business_address"].to_numpy(copy=True)
    pool_names = c_pool["business_name"].to_numpy(copy=True)
    pool_addrs = c_pool["business_address"].to_numpy(copy=True)
    del c_s1, c_pool
    gc.collect()

    # 3. Feature extraction & scoring in batches
    t_sc = time.time()
    country_matches = [[] for _ in range(n_s1)]
    batch_feats = []
    batch_meta = []
    batch_size = 50000

    def score_batch(feats, meta):
        if not feats:
            return
        X = np.array(feats, dtype=np.float32)
        probs = model.predict(X)
        for (s1_idx, cand_id), p in zip(meta, probs):
            if p >= threshold:
                country_matches[s1_idx].append(cand_id)

    @lru_cache(maxsize=100000)
    def get_pool_attrs(pool_pos):
        return compute_attrs(pool_names[pool_pos], pool_addrs[pool_pos])

    print(f"Scoring {total_pairs:,} candidate pairs with LightGBM (threshold={threshold:.2f})...")

    with tqdm(total=total_pairs, desc=f"Scoring {country}", mininterval=2.0) as pbar:
        for s1_idx in range(n_s1):
            s1_attrs = compute_attrs(s1_names[s1_idx], s1_addrs[s1_idx])
            start_pos = offsets[s1_idx]
            end_pos = offsets[s1_idx + 1]
            for rank, pair_pos in enumerate(range(start_pos, end_pos), start=1):
                pool_pos = int(candidate_pool_pos[pair_pos])
                cand_id = pool_ids[pool_pos]
                cand_attrs = get_pool_attrs(pool_pos)
                f = extract_pair_features(
                    s1_attrs[0], s1_attrs[1], s1_attrs[2], s1_attrs[3], s1_attrs[4], s1_attrs[5], s1_attrs[6],
                    cand_attrs[0], cand_attrs[1], cand_attrs[2], cand_attrs[3], cand_attrs[4], cand_attrs[5], cand_attrs[6],
                    float(candidate_scores[pair_pos]), rank
                )
                batch_feats.append(f)
                batch_meta.append((s1_idx, cand_id))

                if len(batch_feats) >= batch_size:
                    score_batch(batch_feats, batch_meta)
                    pbar.update(len(batch_feats))
                    batch_feats = []
                    batch_meta = []

        if batch_feats:
            score_batch(batch_feats, batch_meta)
            pbar.update(len(batch_feats))

    print(f"Scoring complete in {time.time()-t_sc:.2f}s.")

    # 4. Format results keeping Source-1 file order
    match_rows = []
    cand_rows = []
    for s1_idx, s1_id in enumerate(s1_ids):
        start_pos = offsets[s1_idx]
        end_pos = offsets[s1_idx + 1]
        cand_ids = pool_ids[candidate_pool_pos[start_pos:end_pos]]
        cand_str = ",".join(str(cand_id) for cand_id in cand_ids)
        cand_rows.append(f"{s1_id}\t{cand_str}\n")

        unique_matches = list(dict.fromkeys(country_matches[s1_idx]))
        match_rows.append(f"{s1_id}\t{','.join(unique_matches)}\n")

    get_pool_attrs.cache_clear()
    del (
        s1_names,
        s1_addrs,
        pool_names,
        pool_addrs,
        pool_ids,
        candidate_pool_pos,
        candidate_scores,
        offsets,
        s1_ids,
        country_matches,
        batch_feats,
        batch_meta,
    )
    gc.collect()

    return match_rows, cand_rows


def run_full_inference(
    test_dir: str = "dataset/test",
    model_path: str = "models/lgbm_matcher.txt",
    output_dir: str = "output",
    threshold: float = 0.65,
    top_k: int = 35,
    countries: list[str] | None = None,
):
    print("=" * 65)
    print("Amazon ML Challenge 2026: Country-Partitioned Inference Pipeline")
    print("=" * 65)
    t_global_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}.")
    print(f"Loading LightGBM model from {model_path}...")
    model = lgb.Booster(model_file=model_path)

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    countries = countries or ["France", "US", "India"]

    blocker = MultiPassBlocker(top_k=top_k)

    match_out_path = os.path.join(output_dir, "matching_results.tsv")
    cand_out_path = os.path.join(output_dir, "candidate_pairs.tsv")

    with open(match_out_path, "w", encoding="utf-8") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
    with open(cand_out_path, "w", encoding="utf-8") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

    total_entities_written = 0

    for country in countries:
        match_lines, cand_lines = run_country_inference(
            country=country,
            s1_path=s1_path,
            s2_path=s2_path,
            s3_path=s3_path,
            model=model,
            threshold=threshold,
            top_k=top_k,
            blocker=blocker
        )

        with open(match_out_path, "a", encoding="utf-8") as fm:
            fm.writelines(match_lines)
        with open(cand_out_path, "a", encoding="utf-8") as fc:
            fc.writelines(cand_lines)

        total_entities_written += len(match_lines)
        print(f"Appended {len(match_lines):,} rows for {country}. "
              f"Total so far: {total_entities_written:,}")
        del match_lines, cand_lines
        gc.collect()

    total_time = time.time() - t_global_start
    print(f"\n{'='*65}")
    print(f"Inference Completed Successfully in {total_time/60:.2f} minutes!")
    print(f"Total Source 1 entities processed: {total_entities_written:,}")
    print(f"Final output saved to:")
    print(f"  Leaderboard file: {match_out_path}")
    print(f"  Candidate file:   {cand_out_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run country-partitioned inference")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--model-path", default="models/lgbm_matcher.txt")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=35)
    parser.add_argument("--countries", nargs="+", choices=["France", "US", "India"])
    args = parser.parse_args()

    run_full_inference(
        test_dir=args.test_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        top_k=args.top_k,
        countries=args.countries,
    )