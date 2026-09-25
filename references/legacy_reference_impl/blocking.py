"""Multi-pass Inverted Index Blocking for Business Entity Resolution.

Partitions records strictly by country and uses multiple complementary indexing channels:
1. Exact core business name (stripped of legal suffixes)
2. Business name tokens (IDF-weighted)
3. Name character 3-grams (for typo resilience)
4. Address distinctive tokens
5. Address numbers (house numbers, postal/PIN codes)
"""

import os
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
from tqdm import tqdm

from .preprocess import (
    normalize_text,
    get_core_name,
    get_tokens,
    get_char_ngrams,
    get_numbers
)


class MultiPassBlocker:
    def __init__(self, top_k: int = 35, max_token_freq_ratio: float = 0.005):
        self.top_k = top_k
        self.max_token_freq_ratio = max_token_freq_ratio

    def build_index(self, pool_df: pd.DataFrame):
        """Build multi-pass inverted indices for a country pool (S2 + S3)."""
        pool_ids = pool_df["entity_id"].values
        pool_names = pool_df["business_name"].values
        pool_addrs = pool_df["business_address"].values
        n_pool = len(pool_df)

        exact_core_idx = defaultdict(list)
        name_token_idx = defaultdict(list)
        addr_token_idx = defaultdict(list)
        addr_num_idx = defaultdict(list)

        name_freq = defaultdict(int)
        addr_freq = defaultdict(int)

        for idx in range(n_pool):
            norm_name = normalize_text(pool_names[idx])
            core_name = get_core_name(norm_name)
            norm_addr = normalize_text(pool_addrs[idx])

            n_toks = get_tokens(core_name, min_len=3)
            a_toks = get_tokens(norm_addr, min_len=3)
            nums = get_numbers(norm_addr)

            if core_name:
                exact_core_idx[core_name].append(idx)
            for t in n_toks:
                name_freq[t] += 1
            for a in a_toks:
                addr_freq[a] += 1
            for num in nums:
                addr_num_idx[num].append(idx)

        # Thresholds to suppress overly frequent stopwords
        max_n_freq = max(200, int(n_pool * self.max_token_freq_ratio))
        max_a_freq = max(200, int(n_pool * self.max_token_freq_ratio))

        for idx in range(n_pool):
            norm_name = normalize_text(pool_names[idx])
            core_name = get_core_name(norm_name)
            norm_addr = normalize_text(pool_addrs[idx])
            n_toks = get_tokens(core_name, min_len=3)
            a_toks = get_tokens(norm_addr, min_len=3)
            for t in n_toks:
                if name_freq[t] <= max_n_freq:
                    name_token_idx[t].append(idx)
            for a in a_toks:
                if addr_freq[a] <= max_a_freq:
                    addr_token_idx[a].append(idx)

        return {
            "pool_ids": pool_ids,
            "pool_names": pool_names,
            "pool_addrs": pool_addrs,
            "exact_core_idx": exact_core_idx,
            "name_token_idx": name_token_idx,
            "addr_token_idx": addr_token_idx,
            "addr_num_idx": addr_num_idx,
            "name_freq": name_freq,
            "addr_freq": addr_freq,
            "n_pool": n_pool
        }

    def block_entity(
        self,
        s1_core_name: str,
        s1_n_toks: Set[str],
        s1_a_toks: Set[str],
        s1_nums: Set[str],
        idx_data: dict
    ) -> List[Tuple[int, float]]:
        """Score and retrieve top candidate indices from the pool for one S1 record."""
        cand_scores = defaultdict(float)

        exact_core_idx = idx_data["exact_core_idx"]
        name_token_idx = idx_data["name_token_idx"]
        addr_token_idx = idx_data["addr_token_idx"]
        addr_num_idx = idx_data["addr_num_idx"]
        name_freq = idx_data["name_freq"]
        addr_freq = idx_data["addr_freq"]

        # 1. Exact core name match
        if s1_core_name and s1_core_name in exact_core_idx:
            for c_idx in exact_core_idx[s1_core_name]:
                cand_scores[c_idx] += 15.0

        # 2. Distinctive name token match (IDF-weighted)
        for t in s1_n_toks:
            if t in name_token_idx:
                tf = name_freq[t]
                w = 4.0 / (1.0 + tf / 20.0)
                for c_idx in name_token_idx[t]:
                    cand_scores[c_idx] += w

        # 3. Distinctive address token match
        for a in s1_a_toks:
            if a in addr_token_idx:
                af = addr_freq[a]
                w = 3.0 / (1.0 + af / 20.0)
                for c_idx in addr_token_idx[a]:
                    cand_scores[c_idx] += w

        # 4. Address number / pincode match
        for num in s1_nums:
            if num in addr_num_idx and len(addr_num_idx[num]) <= 500:
                w = 2.0 / (1.0 + len(addr_num_idx[num]) / 50.0)
                for c_idx in addr_num_idx[num]:
                    cand_scores[c_idx] += w

        if not cand_scores:
            return []

        # Return sorted list of (pool_idx, score) up to top_k
        if len(cand_scores) <= self.top_k:
            return sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]

    def block_country(
        self,
        s1_df: pd.DataFrame,
        pool_df: pd.DataFrame
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Perform blocking for a single country partition."""
        idx_data = self.build_index(pool_df)
        pool_ids = idx_data["pool_ids"]

        s1_ids = s1_df["entity_id"].values
        s1_names = s1_df["business_name"].values
        s1_addrs = s1_df["business_address"].values

        results = {}
        for i in range(len(s1_df)):
            s1_id = s1_ids[i]
            norm_name = normalize_text(s1_names[i])
            core_name = get_core_name(norm_name)
            norm_addr = normalize_text(s1_addrs[i])

            n_toks = get_tokens(core_name, min_len=3)
            a_toks = get_tokens(norm_addr, min_len=3)
            nums = get_numbers(norm_addr)

            cand_items = self.block_entity(core_name, n_toks, a_toks, nums, idx_data)
            results[s1_id] = [(pool_ids[c_idx], score) for c_idx, score in cand_items]

        return results

    def block_all(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Partition by country and execute blocking across all countries."""
        countries = s1_df["country"].unique()
        all_candidates = {}

        for country in countries:
            print(f"Blocking country: {country}...")
            c_s1 = s1_df[s1_df["country"] == country]
            c_s2 = s2_df[s2_df["country"] == country]
            c_s3 = s3_df[s3_df["country"] == country]
            pool = pd.concat([c_s2, c_s3], ignore_index=True)

            print(f"  {country}: {len(c_s1):,} S1 entities against {len(pool):,} pool records.")
            c_res = self.block_country(c_s1, pool)
            all_candidates.update(c_res)

        return all_candidates


def save_candidate_pairs_tsv(
    candidates_dict: Dict[str, List[Tuple[str, float]]],
    output_path: str,
    all_s1_ids: List[str]
):
    """Write candidate pairs to candidate_pairs.tsv ensuring all S1 IDs are present."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = candidates_dict.get(s1_id, [])
            cand_str = ",".join(c_id for c_id, _ in cands)
            f.write(f"{s1_id}\t{cand_str}\n")
    print(f"Saved candidate pairs for {len(all_s1_ids):,} entities to {output_path}")
