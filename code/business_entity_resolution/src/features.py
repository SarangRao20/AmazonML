"""Pairwise Feature Engineering using RapidFuzz C++ engine and token similarities."""

from typing import List, Dict, Any, Tuple
import numpy as np
import rapidfuzz.fuzz as fuzz

from .preprocess import (
    normalize_text,
    get_core_name,
    get_tokens,
    get_char_ngrams,
    get_numbers
)

FEATURE_NAMES = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_core_ratio",
    "name_token_jaccard",
    "name_char3_jaccard",
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_token_jaccard",
    "addr_num_jaccard",
    "addr_num_match",
    "addr_empty",
    "blocking_score",
    "blocking_rank",
    "combined_token_set_ratio"
]


def extract_pair_features(
    s1_norm_name: str,
    s1_core_name: str,
    s1_norm_addr: str,
    s1_n_toks: set,
    s1_a_toks: set,
    s1_nums: set,
    s1_c3: set,
    m_norm_name: str,
    m_core_name: str,
    m_norm_addr: str,
    m_n_toks: set,
    m_a_toks: set,
    m_nums: set,
    m_c3: set,
    blocking_score: float,
    blocking_rank: int
) -> List[float]:
    """Compute feature vector for a single (S1, S2/S3) pair."""
    # 1. Name features
    n_rat = fuzz.ratio(s1_norm_name, m_norm_name) / 100.0
    n_part = fuzz.partial_ratio(s1_norm_name, m_norm_name) / 100.0
    n_sort = fuzz.token_sort_ratio(s1_norm_name, m_norm_name) / 100.0
    n_set = fuzz.token_set_ratio(s1_norm_name, m_norm_name) / 100.0
    n_core = fuzz.ratio(s1_core_name, m_core_name) / 100.0

    n_jacc = len(s1_n_toks & m_n_toks) / len(s1_n_toks | m_n_toks) if (s1_n_toks or m_n_toks) else 0.0
    n_c3_jacc = len(s1_c3 & m_c3) / len(s1_c3 | m_c3) if (s1_c3 or m_c3) else 0.0

    # 2. Address features
    addr_empty = 1.0 if not m_norm_addr else 0.0
    if not m_norm_addr:
        a_rat = 0.0
        a_sort = 0.0
        a_set = 0.0
        a_jacc = 0.0
        num_jacc = 0.0
        num_match = 0.0
    else:
        a_rat = fuzz.ratio(s1_norm_addr, m_norm_addr) / 100.0
        a_sort = fuzz.token_sort_ratio(s1_norm_addr, m_norm_addr) / 100.0
        a_set = fuzz.token_set_ratio(s1_norm_addr, m_norm_addr) / 100.0
        a_jacc = len(s1_a_toks & m_a_toks) / len(s1_a_toks | m_a_toks) if (s1_a_toks or m_a_toks) else 0.0
        num_jacc = len(s1_nums & m_nums) / len(s1_nums | m_nums) if (s1_nums or m_nums) else 0.0
        num_match = 1.0 if (s1_nums & m_nums) else 0.0

    # 3. Combined text ratio
    s1_comb = f"{s1_norm_name} {s1_norm_addr}"
    m_comb = f"{m_norm_name} {m_norm_addr}"
    comb_set = fuzz.token_set_ratio(s1_comb, m_comb) / 100.0

    return [
        n_rat,
        n_part,
        n_sort,
        n_set,
        n_core,
        n_jacc,
        n_c3_jacc,
        a_rat,
        a_sort,
        a_set,
        a_jacc,
        num_jacc,
        num_match,
        addr_empty,
        blocking_score,
        float(blocking_rank),
        comb_set
    ]
