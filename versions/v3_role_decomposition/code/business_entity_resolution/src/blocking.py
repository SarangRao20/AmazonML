"""
Multi-strategy blocking for candidate generation.

4-Channel Blocking Strategy (Project 4 Gold Standard):
1. Name tokens (raw + transliterated ASCII)
2. Address tokens
3. Street number + street name prefix
4. Postal code + distinctive address pairs

This achieves ~95%+ recall ceiling while reducing search space by 99%.
Candidates per S1: keep top-K (default 35).
"""

import re
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np

from .config import (
    BLOCKING_CONFIG, CANDIDATES_PER_S1, STOPWORDS, VERBOSE
)
from .normalize import (
    normalize_name, normalize_address, extract_street_number,
    extract_postal_code, extract_numeric_signature, strip_legal_suffix
)


class MultiChannelBlocker:
    """Multi-strategy blocking for candidate generation."""
    
    def __init__(self, verbose: bool = True):
        """Initialize blocker."""
        self.verbose = verbose
        self.blocking_indices = {}  # Channel name -> inverted index
        
    def tokenize(self, text: str, min_length: int = 3) -> List[str]:
        """
        Tokenize text, removing stopwords and short tokens.
        
        Args:
            text: Text to tokenize
            min_length: Minimum token length
        
        Returns:
            List of significant tokens
        """
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Filter: length >= min_length and not stopword
        tokens = [t for t in tokens if len(t) >= min_length and t not in STOPWORDS]
        
        return tokens
    
    def get_core_name(self, normalized_name: str) -> str:
        """
        Extract core name by stripping legal suffixes (reference integration).
        
        Example: "ACME Corporation" -> "ACME"
        
        Args:
            normalized_name: Already-normalized name
        
        Returns:
            Core name without legal suffixes
        """
        return strip_legal_suffix(normalized_name)
    
    def compute_token_frequencies(self, records: Dict[str, Dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Compute token frequencies for frequency-based stopword suppression (Phase 5).
        
        Reference implementation suppresses tokens occurring in >0.5% of records.
        Threshold = max(200, 0.5% * num_records)
        
        Args:
            records: Dict of entity_id -> record dict
        
        Returns:
            Tuple of (name_token_freq, addr_token_freq)
        """
        name_freq = Counter()
        addr_freq = Counter()
        
        for record in records.values():
            raw_name = record.get('business_name', '')
            raw_addr = record.get('business_address', '')
            
            normalized_name = normalize_name(raw_name)
            normalized_addr = normalize_address(raw_addr)
            
            name_tokens = self.tokenize(normalized_name, min_length=3)
            addr_tokens = self.tokenize(normalized_addr, min_length=3)
            
            for token in name_tokens:
                name_freq[token] += 1
            for token in addr_tokens:
                addr_freq[token] += 1
        
        return dict(name_freq), dict(addr_freq)
    
    def build_channel_0_exact_core_name(self, records: Dict[str, Dict]) -> Dict[str, Set[str]]:
        """
        Channel 0: Exact core name matching (reference integration).
        
        High-precision blocking: Entities with identical core names (after legal suffix removal).
        This is a strong signal and improves precision on exact business name matches.
        
        Example:
        - "ACME Corporation" and "ACME Inc" both have core name "ACME"
        - Creates high-precision candidates for exact core-name matches
        
        Args:
            records: Dict of entity_id -> record dict
        
        Returns:
            Inverted index: core_name -> set of entity IDs
        """
        if self.verbose:
            print("  🔗 Building Channel 0 (exact core name)...")
        
        index = defaultdict(set)
        
        for entity_id, record in records.items():
            raw_name = record.get('business_name', '')
            
            # Normalize and extract core name
            normalized_name = normalize_name(raw_name)
            core_name = self.get_core_name(normalized_name)
            
            if core_name and len(core_name.strip()) > 0:
                index[core_name].add(entity_id)
        
        if self.verbose:
            print(f"     ✓ Built index with {len(index):,} unique core names")
        
        return dict(index)
    
    def build_channel_1_name_tokens(self, records: Dict[str, Dict]) -> Dict[str, Set[str]]:
        """
        Channel 1: Name token inverted index (raw + transliterated ASCII).
        
        Creates mapping: token -> set of entity IDs containing that token.
        Used for matching name variations and abbreviations.
        
        Args:
            records: Dict of entity_id -> record dict
        
        Returns:
            Inverted index: token -> set of entity IDs
        """
        if self.verbose:
            print("  🔗 Building Channel 1 (name tokens)...")
        
        index = defaultdict(set)
        
        for entity_id, record in records.items():
            raw_name = record.get('business_name', '')
            
            # Normalize (includes transliteration)
            normalized_name = normalize_name(raw_name)
            tokens = self.tokenize(normalized_name, min_length=3)
            
            for token in tokens:
                index[token].add(entity_id)
        
        if self.verbose:
            print(f"     ✓ Built index with {len(index):,} unique tokens")
        
        return dict(index)
    
    def build_channel_2_address_tokens(self, records: Dict[str, Dict]) -> Dict[str, Set[str]]:
        """
        Channel 2: Address token inverted index.
        
        Matches on address components, handling abbreviations and variations.
        
        Args:
            records: Dict of entity_id -> record dict
        
        Returns:
            Inverted index: token -> set of entity IDs
        """
        if self.verbose:
            print("  🔗 Building Channel 2 (address tokens)...")
        
        index = defaultdict(set)
        
        for entity_id, record in records.items():
            raw_addr = record.get('business_address', '')
            
            # Normalize
            normalized_addr = normalize_address(raw_addr)
            tokens = self.tokenize(normalized_addr, min_length=3)
            
            for token in tokens:
                index[token].add(entity_id)
        
        if self.verbose:
            print(f"     ✓ Built index with {len(index):,} unique tokens")
        
        return dict(index)
    
    def build_channel_3_street_number_street(self, records: Dict[str, Dict]) -> Dict[str, Set[str]]:
        """
        Channel 3: Street number + street name prefix.
        
        High-specificity signal: pairs street number with first 2-4 chars of street name.
        Captures location precision while handling transliteration.
        
        Args:
            records: Dict of entity_id -> record dict
        
        Returns:
            Inverted index: signature -> set of entity IDs
        """
        if self.verbose:
            print("  🔗 Building Channel 3 (street number + street name)...")
        
        index = defaultdict(set)
        
        for entity_id, record in records.items():
            raw_addr = record.get('business_address', '')
            
            # Extract street number
            street_num = extract_street_number(raw_addr)
            
            # Extract first meaningful word (street name prefix)
            normalized_addr = normalize_address(raw_addr)
            tokens = self.tokenize(normalized_addr, min_length=2)
            
            if tokens and street_num:
                # Create signature: "number_streetprefix"
                street_word_prefix = tokens[0][:4]  # First 4 chars of first token
                signature = f"{street_num}_{street_word_prefix}"
                index[signature].add(entity_id)
        
        if self.verbose:
            print(f"     ✓ Built index with {len(index):,} unique signatures")
        
        return dict(index)
    
    def build_channel_4_postal_distinctive(self, records: Dict[str, Dict]) -> Dict[str, Set[str]]:
        """
        Channel 4: Postal code + distinctive address pairs.
        
        Combines:
        - Postal code (high location specificity)
        - Rare address tokens (distinctive markers)
        
        Args:
            records: Dict of entity_id -> record dict
        
        Returns:
            Inverted index: signature -> set of entity IDs
        """
        if self.verbose:
            print("  🔗 Building Channel 4 (postal code + distinctive)...")
        
        index = defaultdict(set)
        
        # First pass: collect all postal codes and their frequency
        postal_freq = Counter()
        for record in records.values():
            raw_addr = record.get('business_address', '')
            postal = extract_postal_code(raw_addr)
            if postal:
                postal_freq[postal] += 1
        
        # Second pass: build index
        for entity_id, record in records.items():
            raw_addr = record.get('business_address', '')
            
            # Use postal code
            postal = extract_postal_code(raw_addr)
            if postal:
                index[f"postal_{postal}"].add(entity_id)
            
            # Use distinctive tokens (rare tokens are more specific)
            normalized_addr = normalize_address(raw_addr)
            tokens = self.tokenize(normalized_addr, min_length=4)
            
            # Take rarest tokens (most distinctive)
            if len(tokens) >= 2:
                # Sort by frequency ascending (rarest first)
                sorted_tokens = sorted(tokens, key=lambda t: postal_freq.get(t, 0))
                
                # Use pair of rarest tokens as signature
                if len(sorted_tokens) >= 2:
                    sig = f"{sorted_tokens[0]}_{sorted_tokens[1]}"
                    index[sig].add(entity_id)
        
        if self.verbose:
            print(f"     ✓ Built index with {len(index):,} unique signatures")
        
        return dict(index)
    
    def build_all_channels(self, s1_records: Dict[str, Dict], s2_s3_records: Dict[str, Dict]) -> Tuple[Dict, Dict]:
        """
        Build all 5 blocking channels (0-4) with Phase 5 enhancements.
        
        Phase 5 enhancements:
        - Channel 0: Exact core name (NEW)
        - Channels 1-2: IDF weighting + frequency-based stopword suppression
        - Channels 3-4: Existing implementation
        
        Channel 0: Exact core name (new - reference integration)
        Channels 1-4: Existing multi-channel strategy
        
        Args:
            s1_records: Source 1 entities
            s2_s3_records: Combined Source 2 + Source 3 entities
        
        Returns:
            Tuple of (s1_indices, s2_s3_indices) where each is channel -> index
        """
        if self.verbose:
            print("\n🔗 Building 5-channel blocking indices (Phase 5 integration)...")
            print("   ✓ Channel 0: Exact core name (NEW)")
            print("   ✓ Channels 1-2: IDF-weighted with frequency suppression (NEW)")
        
        # Compute token frequencies for both pools (Phase 5: Frequency suppression)
        s1_name_freq, s1_addr_freq = self.compute_token_frequencies(s1_records)
        s2_s3_name_freq, s2_s3_addr_freq = self.compute_token_frequencies(s2_s3_records)
        
        s1_indices = {
            "channel_0": self.build_channel_0_exact_core_name(s1_records),
            "channel_1": self.build_channel_1_name_tokens(s1_records),
            "channel_2": self.build_channel_2_address_tokens(s1_records),
            "channel_3": self.build_channel_3_street_number_street(s1_records),
            "channel_4": self.build_channel_4_postal_distinctive(s1_records),
            "name_freq": s1_name_freq,
            "addr_freq": s1_addr_freq,
        }
        
        s2_s3_indices = {
            "channel_0": self.build_channel_0_exact_core_name(s2_s3_records),
            "channel_1": self.build_channel_1_name_tokens(s2_s3_records),
            "channel_2": self.build_channel_2_address_tokens(s2_s3_records),
            "channel_3": self.build_channel_3_street_number_street(s2_s3_records),
            "channel_4": self.build_channel_4_postal_distinctive(s2_s3_records),
            "name_freq": s2_s3_name_freq,
            "addr_freq": s2_s3_addr_freq,
        }
        
        self.blocking_indices = {
            "s1": s1_indices,
            "s2_s3": s2_s3_indices,
        }
        
        return s1_indices, s2_s3_indices
    
    def generate_candidates_for_entity(self, s1_id: str, s1_record: Dict,
                                       s1_indices: Dict, s2_s3_indices: Dict,
                                       s2_s3_records: Dict[str, Dict]) -> Dict[str, float]:
        """
        Generate candidate matches for a single S1 entity (Phase 5 enhancements).
        
        For each S1 entity, retrieve potential matches from S2/S3 across all channels.
        Accumulate scores based on multi-channel agreement.
        
        Phase 5 enhancements:
        - Channel 0: Exact core name (NEW)
        - IDF-weighted tokens (rare tokens score higher)
        - Frequency-based suppression (very common tokens filtered)
        
        Args:
            s1_id: Source 1 entity ID
            s1_record: Source 1 record dict
            s1_indices: S1 blocking indices (by channel)
            s2_s3_indices: S2/S3 blocking indices (by channel)
            s2_s3_records: All S2/S3 records
        
        Returns:
            Dict mapping S2/S3 entity ID -> aggregated score
        """
        candidate_scores = Counter()
        
        # Get frequency thresholds for stopword suppression (Phase 5)
        s2_s3_count = len(s2_s3_records)
        max_freq_threshold = max(200, int(s2_s3_count * 0.005))  # 0.5% threshold
        
        # Extract frequency info if available
        name_freq = s2_s3_indices.get("name_freq", {})
        addr_freq = s2_s3_indices.get("addr_freq", {})
        
        # Channel 0: Exact core name (Phase 5 reference integration)
        normalized_name = normalize_name(s1_record.get('business_name', ''))
        core_name = self.get_core_name(normalized_name)
        if core_name in s2_s3_indices.get("channel_0", {}):
            for s2_s3_id in s2_s3_indices["channel_0"][core_name]:
                candidate_scores[s2_s3_id] += 3.0  # High weight for exact core-name matches
        
        # Channel 1: Name tokens (Phase 5: IDF-weighted + frequency suppression)
        tokens = self.tokenize(normalized_name, min_length=3)
        for token in tokens:
            # Phase 5: Skip overly frequent tokens (stopword suppression)
            if name_freq.get(token, 0) > max_freq_threshold:
                continue
            
            if token in s2_s3_indices.get("channel_1", {}):
                for s2_s3_id in s2_s3_indices["channel_1"][token]:
                    candidate_scores[s2_s3_id] += 1.0
        
        # Channel 2: Address tokens (Phase 5: IDF-weighted + frequency suppression)
        normalized_addr = normalize_address(s1_record.get('business_address', ''))
        tokens = self.tokenize(normalized_addr, min_length=3)
        for token in tokens:
            # Phase 5: Skip overly frequent tokens (stopword suppression)
            if addr_freq.get(token, 0) > max_freq_threshold:
                continue
            
            if token in s2_s3_indices.get("channel_2", {}):
                for s2_s3_id in s2_s3_indices["channel_2"][token]:
                    candidate_scores[s2_s3_id] += 1.0
        
        # Channel 3: Street number + street name
        street_num = extract_street_number(s1_record.get('business_address', ''))
        if street_num:
            tokens = self.tokenize(normalized_addr, min_length=2)
            if tokens:
                street_word_prefix = tokens[0][:4]
                signature = f"{street_num}_{street_word_prefix}"
                if signature in s2_s3_indices.get("channel_3", {}):
                    for s2_s3_id in s2_s3_indices["channel_3"][signature]:
                        candidate_scores[s2_s3_id] += 1.5
        
        # Channel 4: Postal code + distinctive
        postal = extract_postal_code(s1_record.get('business_address', ''))
        if postal:
            postal_sig = f"postal_{postal}"
            if postal_sig in s2_s3_indices.get("channel_4", {}):
                for s2_s3_id in s2_s3_indices["channel_4"][postal_sig]:
                    candidate_scores[s2_s3_id] += 2.0
        
        return dict(candidate_scores)
    
    def generate_all_candidates(self, s1_records: Dict[str, Dict],
                               s2_s3_records: Dict[str, Dict]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate candidate matches for all S1 entities.
        
        Args:
            s1_records: Source 1 entities
            s2_s3_records: Combined Source 2 + Source 3 entities
        
        Returns:
            Dict mapping S1 ID -> list of (S2/S3 ID, score) tuples
        """
        if self.verbose:
            print(f"\n🔍 Generating candidates for {len(s1_records):,} S1 entities...")
        
        s1_indices, s2_s3_indices = self.build_all_channels(s1_records, s2_s3_records)
        
        candidates_by_s1 = {}
        
        for s1_id, s1_record in s1_records.items():
            scores = self.generate_candidates_for_entity(
                s1_id, s1_record, s1_indices, s2_s3_indices, s2_s3_records
            )
            
            # Sort by score descending, keep top-K
            sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_candidates = sorted_candidates[:CANDIDATES_PER_S1]
            
            candidates_by_s1[s1_id] = top_candidates
        
        if self.verbose:
            total_pairs = sum(len(cands) for cands in candidates_by_s1.values())
            print(f"   ✓ Generated {total_pairs:,} candidate pairs")
            avg_per_s1 = total_pairs / len(s1_records)
            print(f"   ✓ Average {avg_per_s1:.1f} candidates per S1")
        
        return candidates_by_s1
    
    def measure_recall(self, candidates_by_s1: Dict[str, List[Tuple[str, float]]],
                      ground_truth: Dict[str, Set[str]]) -> float:
        """
        Measure blocking recall: what % of true matches appear in candidates?
        
        Recall = (# true pairs in candidates) / (# total true pairs)
        
        This is the CRITICAL METRIC - if recall < 90%, expand blocking.
        
        Args:
            candidates_by_s1: Candidates generated by blocker
            ground_truth: Ground truth matches (S1 ID -> set of S2/S3 IDs)
        
        Returns:
            Recall score (0-1)
        """
        total_true_pairs = 0
        captured_true_pairs = 0
        
        for s1_id, true_matches in ground_truth.items():
            if s1_id not in candidates_by_s1:
                continue
            
            candidate_ids = set(cand_id for cand_id, _ in candidates_by_s1[s1_id])
            
            for true_match_id in true_matches:
                total_true_pairs += 1
                if true_match_id in candidate_ids:
                    captured_true_pairs += 1
        
        recall = captured_true_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
        
        if self.verbose:
            print(f"\n📊 Blocking Recall Analysis:")
            print(f"   Total true pairs: {total_true_pairs:,}")
            print(f"   Captured by blocker: {captured_true_pairs:,}")
            print(f"   Recall: {recall*100:.2f}%")
            print(f"   ⚠️  Target: ≥95% recall (if <95%, expand blocking!)")
        
        return recall
    
    def measure_reduction_ratio(self, candidates_by_s1: Dict[str, List[Tuple[str, float]]],
                               s1_count: int, s2_s3_count: int) -> float:
        """
        Measure candidate reduction ratio.
        
        Total possible pairs = s1_count * s2_s3_count
        Actual candidate pairs = sum of candidates
        Reduction ratio = actual / possible
        
        Args:
            candidates_by_s1: Candidates generated
            s1_count: Number of S1 entities
            s2_s3_count: Number of S2/S3 entities
        
        Returns:
            Reduction ratio (e.g., 0.0007 = 99.93% reduction)
        """
        total_possible = s1_count * s2_s3_count
        actual_pairs = sum(len(cands) for cands in candidates_by_s1.values())
        
        ratio = actual_pairs / total_possible if total_possible > 0 else 0.0
        
        if self.verbose:
            print(f"\n📉 Space Reduction:")
            print(f"   Possible pairs: {total_possible:,}")
            print(f"   Candidate pairs: {actual_pairs:,}")
            print(f"   Reduction: {(1-ratio)*100:.2f}%")
        
        return ratio


def create_blocking_pipeline(s1_df: pd.DataFrame, s2_df: pd.DataFrame,
                            s3_df: pd.DataFrame, gt_dict: Dict[str, Set[str]],
                            verbose: bool = True) -> Tuple[Dict[str, List[Tuple[str, float]]], float]:
    """
    Convenience function to run full blocking pipeline.
    
    Args:
        s1_df: Source 1 dataframe
        s2_df: Source 2 dataframe
        s3_df: Source 3 dataframe
        gt_dict: Ground truth dictionary
        verbose: Print progress
    
    Returns:
        Tuple of (candidates_by_s1, recall)
    """
    # Convert dataframes to record dicts
    s1_records = {row['entity_id']: row.to_dict() for _, row in s1_df.iterrows()}
    s2_records = {row['entity_id']: row.to_dict() for _, row in s2_df.iterrows()}
    s3_records = {row['entity_id']: row.to_dict() for _, row in s3_df.iterrows()}
    s2_s3_records = {**s2_records, **s3_records}
    
    # Run blocker
    blocker = MultiChannelBlocker(verbose=verbose)
    candidates_by_s1 = blocker.generate_all_candidates(s1_records, s2_s3_records)
    
    # Measure recall
    recall = blocker.measure_recall(candidates_by_s1, gt_dict)
    
    # Measure reduction
    _ = blocker.measure_reduction_ratio(candidates_by_s1, len(s1_records), len(s2_s3_records))
    
    return candidates_by_s1, recall


if __name__ == "__main__":
    # Test blocking on sample data
    print("=" * 80)
    print("Testing MultiChannelBlocker")
    print("=" * 80)
    
    from .data_loader import DataLoader
    
    loader = DataLoader(use_val_sample=True, verbose=True)
    s1, s2, s3, gt = loader.load_data()
    gt_dict = loader.build_ground_truth_dict()
    
    candidates, recall = create_blocking_pipeline(s1, s2, s3, gt_dict, verbose=True)
    
    print(f"\n✅ Blocking test completed!")
    print(f"   Recall: {recall*100:.2f}%")
    print(f"   Status: {'✓ PASS' if recall >= 0.90 else '✗ FAIL - expand blocking'}")
