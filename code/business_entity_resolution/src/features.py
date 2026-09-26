"""
Feature engineering for entity matching.

51 Advanced Features (Phase 2) including:
- 15 name similarity metrics (raw + transliterated + first-token + structural)
- 15 address similarity metrics (same structure)
- 1 number overlap (Project 2 discovery: +0.74 impact!)
- 2 token-level similarities
- 2 exact match flags
- 2 structural features
- 2 legal suffix indicators (NEW)
- 7 composite/interaction features (NEW)
- 4 postal code hierarchical features (NEW)
- 2 landmark features (NEW)
- 19 multi-channel meta-features (NEW - Task 4)

Multi-channel features track blocking performance per channel:
- Per-channel indicators (8): name_tokens, addr_tokens, street_number, postal
- Best-rank features (4): best_channel_rank, rank_agreement, reciprocal_rank, channel_count
- Agreement features (7): channels_agree, name_addr_agreement, top_channel, channel_diversity, cross_channel_rank, avg_channel_rank, channel_confidence

RapidFuzz achieves >11,800 pairs/sec on C++ backend.
Phase 2 adds +34 features for 97-98% F₀.₅ target.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter

try:
    from rapidfuzz import fuzz, distance
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    print("⚠️  RapidFuzz not installed. Using basic string similarity fallback.")
    RAPIDFUZZ_AVAILABLE = False

from .config import FEATURE_NAMES, NUM_FEATURES, VERBOSE
from .normalize import normalize_name, normalize_address, extract_numeric_signature
from collections.abc import Mapping


def get_char_ngrams(text: str, n: int = 3) -> set:
    """
    Extract character n-grams from text (reference implementation).
    
    Example: "john" with n=3 -> {"joh", "ohn"}
    
    Args:
        text: Input string (spaces removed)
        n: N-gram size (default 3)
    
    Returns:
        Set of character n-grams
    """
    if not text or len(text) < n:
        return {text} if text else set()
    cleaned = text.replace(' ', '')
    return {cleaned[i:i+n] for i in range(len(cleaned) - n + 1)}


class FeatureExtractor:
    """Extract matching features for candidate pairs."""
    
    def __init__(self, verbose: bool = False):
        """Initialize feature extractor."""
        self.verbose = verbose
    
    def extract_name_similarity_features(self, name1: str, name2: str) -> Dict[str, float]:
        """
        Extract 5 name similarity features using RapidFuzz.
        
        Features:
        - ratio: Overall string similarity
        - partial_ratio: Best alignment (handles substrings)
        - token_sort_ratio: Handles reordering
        - token_set_ratio: Handles extra/missing tokens
        - jaro_winkler: Phonetic-friendly, errors at start weighted more
        
        Args:
            name1: First name (normalized)
            name2: Second name (normalized)
        
        Returns:
            Dict with 5 name similarity features
        """
        if not name1 or not name2:
            return {
                "name_ratio": 0.0,
                "name_partial_ratio": 0.0,
                "name_token_sort_ratio": 0.0,
                "name_token_set_ratio": 0.0,
                "name_jaro_winkler": 0.0,
            }
        
        if RAPIDFUZZ_AVAILABLE:
            return {
                "name_ratio": fuzz.ratio(name1, name2) / 100.0,
                "name_partial_ratio": fuzz.partial_ratio(name1, name2) / 100.0,
                "name_token_sort_ratio": fuzz.token_sort_ratio(name1, name2) / 100.0,
                "name_token_set_ratio": fuzz.token_set_ratio(name1, name2) / 100.0,
                "name_jaro_winkler": distance.JaroWinkler.normalized_similarity(name1, name2),
            }
        else:
            # Fallback: basic similarity
            ratio = SequenceMatcher(None, name1, name2).ratio()
            return {
                "name_ratio": ratio,
                "name_partial_ratio": ratio,
                "name_token_sort_ratio": ratio,
                "name_token_set_ratio": ratio,
                "name_jaro_winkler": ratio,
            }
    
    def extract_address_similarity_features(self, addr1: str, addr2: str) -> Dict[str, float]:
        """
        Extract 5 address similarity features.
        
        Same metrics as name, applied to addresses.
        
        Args:
            addr1: First address (normalized)
            addr2: Second address (normalized)
        
        Returns:
            Dict with 5 address similarity features
        """
        if not addr1 or not addr2:
            return {
                "addr_ratio": 0.0,
                "addr_partial_ratio": 0.0,
                "addr_token_sort_ratio": 0.0,
                "addr_token_set_ratio": 0.0,
                "addr_jaro_winkler": 0.0,
            }
        
        if RAPIDFUZZ_AVAILABLE:
            return {
                "addr_ratio": fuzz.ratio(addr1, addr2) / 100.0,
                "addr_partial_ratio": fuzz.partial_ratio(addr1, addr2) / 100.0,
                "addr_token_sort_ratio": fuzz.token_sort_ratio(addr1, addr2) / 100.0,
                "addr_token_set_ratio": fuzz.token_set_ratio(addr1, addr2) / 100.0,
                "addr_jaro_winkler": distance.JaroWinkler.normalized_similarity(addr1, addr2),
            }
        else:
            ratio = SequenceMatcher(None, addr1, addr2).ratio()
            return {
                "addr_ratio": ratio,
                "addr_partial_ratio": ratio,
                "addr_token_sort_ratio": ratio,
                "addr_token_set_ratio": ratio,
                "addr_jaro_winkler": ratio,
            }
    
    def extract_name_ngram_jaccard(self, name1: str, name2: str) -> Dict[str, float]:
        """
        Extract character 3-gram Jaccard similarity for names (reference integration).
        
        This feature improves typo tolerance by matching character n-grams.
        Example: "john" vs "joni" both have {joh, ohn, oni} with high overlap.
        
        Args:
            name1: First name (normalized)
            name2: Second name (normalized)
        
        Returns:
            Dict with 1 feature: name_char3_jaccard
        """
        if not name1 or not name2:
            return {"name_char3_jaccard": 0.0}
        
        ngrams1 = get_char_ngrams(name1, n=3)
        ngrams2 = get_char_ngrams(name2, n=3)
        
        if not ngrams1 or not ngrams2:
            return {"name_char3_jaccard": 0.0}
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        jaccard = intersection / union if union > 0 else 0.0
        return {"name_char3_jaccard": jaccard}
    
    def extract_address_ngram_jaccard(self, addr1: str, addr2: str) -> Dict[str, float]:
        """
        Extract character 3-gram Jaccard similarity for addresses (reference integration).
        
        Same logic as name 3-grams, applied to addresses for typo tolerance.
        
        Args:
            addr1: First address (normalized)
            addr2: Second address (normalized)
        
        Returns:
            Dict with 1 feature: addr_char3_jaccard
        """
        if not addr1 or not addr2:
            return {"addr_char3_jaccard": 0.0}
        
        ngrams1 = get_char_ngrams(addr1, n=3)
        ngrams2 = get_char_ngrams(addr2, n=3)
        
        if not ngrams1 or not ngrams2:
            return {"addr_char3_jaccard": 0.0}
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        jaccard = intersection / union if union > 0 else 0.0
        return {"addr_char3_jaccard": jaccard}
    
    def extract_number_overlap_feature(self, addr1: str, addr2: str) -> float:
        """
        Extract numeric signature overlap (Project 2 discovery).
        
        This feature had +0.74 impact on F₀.₅ in Project 2!
        
        Extracts all numbers from both addresses, computes Jaccard similarity.
        Handles street numbers, suite numbers, ZIP codes.
        
        Example:
        - "456 Main Street, Suite 789" -> ["456", "789"]
        - "456 Main Lane, Apt 789" -> ["456", "789"]
        - Overlap = 2/2 = 1.0 (perfect match)
        
        Args:
            addr1: First address
            addr2: Second address
        
        Returns:
            Numeric overlap score (0-1)
        """
        nums1 = set(extract_numeric_signature(addr1).split())
        nums2 = set(extract_numeric_signature(addr2).split())
        
        if not nums1 and not nums2:
            return 1.0  # Both have no numbers (match)
        if not nums1 or not nums2:
            return 0.0  # One has numbers, other doesn't (mismatch)
        
        # Jaccard similarity of numeric sets
        intersection = len(nums1 & nums2)
        union = len(nums1 | nums2)
        
        return intersection / union if union > 0 else 0.0
    
    def extract_token_level_features(self, name1: str, name2: str, 
                                     addr1: str, addr2: str) -> Dict[str, float]:
        """
        Extract token-level similarity features.
        
        Features:
        - token_overlap_count: # common tokens (name + addr)
        - token_jaccard_sim: Jaccard similarity of token sets
        
        Args:
            name1, name2: Normalized names
            addr1, addr2: Normalized addresses
        
        Returns:
            Dict with 2 token features
        """
        # Tokenize names
        tokens1_name = set(name1.lower().split())
        tokens2_name = set(name2.lower().split())
        
        # Tokenize addresses
        tokens1_addr = set(addr1.lower().split())
        tokens2_addr = set(addr2.lower().split())
        
        # Combine
        tokens1_all = tokens1_name | tokens1_addr
        tokens2_all = tokens2_name | tokens2_addr
        
        # Overlap count
        overlap = len(tokens1_all & tokens2_all)
        
        # Jaccard similarity
        if len(tokens1_all | tokens2_all) > 0:
            jaccard = len(tokens1_all & tokens2_all) / len(tokens1_all | tokens2_all)
        else:
            jaccard = 1.0
        
        return {
            "token_overlap_count": float(overlap),
            "token_jaccard_sim": jaccard,
        }
    
    def extract_exact_match_features(self, name1: str, name2: str,
                                     addr1: str, addr2: str) -> Dict[str, float]:
        """
        Extract exact match indicator features.
        
        Features:
        - name_exact_match: 1.0 if names identical, 0.0 otherwise
        - addr_exact_match: 1.0 if addresses identical, 0.0 otherwise
        
        Args:
            name1, name2: Normalized names
            addr1, addr2: Normalized addresses
        
        Returns:
            Dict with 2 exact match flags
        """
        return {
            "name_exact_match": 1.0 if name1.strip() == name2.strip() else 0.0,
            "addr_exact_match": 1.0 if addr1.strip() == addr2.strip() else 0.0,
        }
    
    def extract_structural_features(self, name1: str, name2: str,
                                    addr1: str, addr2: str) -> Dict[str, float]:
        """
        Extract structural features.
        
        Features:
        - name_length_ratio: Ratio of name lengths (0-1)
        - addr_length_ratio: Ratio of address lengths (0-1)
        
        Args:
            name1, name2: Normalized names
            addr1, addr2: Normalized addresses
        
        Returns:
            Dict with 2 structural features
        """
        # Length ratios (bounded to [0, 1])
        name_len1 = len(name1)
        name_len2 = len(name2)
        
        if max(name_len1, name_len2) > 0:
            name_length_ratio = min(name_len1, name_len2) / max(name_len1, name_len2)
        else:
            name_length_ratio = 1.0
        
        addr_len1 = len(addr1)
        addr_len2 = len(addr2)
        
        if max(addr_len1, addr_len2) > 0:
            addr_length_ratio = min(addr_len1, addr_len2) / max(addr_len1, addr_len2)
        else:
            addr_length_ratio = 1.0
        
        return {
            "name_length_ratio": name_length_ratio,
            "addr_length_ratio": addr_length_ratio,
        }
    
    def extract_legal_suffix_features(self, name1: str, name2: str) -> Dict[str, float]:
        """
        Extract legal suffix indicator features (Phase 2).
        
        Features:
        - both_have_suffix: 1.0 if both names have legal suffix
        - suffix_match: 1.0 if suffixes match exactly
        
        Args:
            name1, name2: Normalized names
        
        Returns:
            Dict with 2 legal suffix features
        """
        from .normalize import strip_legal_suffix
        
        # Check if names have suffixes
        root1 = strip_legal_suffix(name1)
        root2 = strip_legal_suffix(name2)
        
        has_suffix1 = len(root1) < len(name1)
        has_suffix2 = len(root2) < len(name2)
        
        both_have = 1.0 if (has_suffix1 and has_suffix2) else 0.0
        
        # Check if suffixes match
        suffix1 = name1[len(root1):].strip() if has_suffix1 else ""
        suffix2 = name2[len(root2):].strip() if has_suffix2 else ""
        suffix_match = 1.0 if (suffix1 and suffix2 and suffix1 == suffix2) else 0.0
        
        return {
            "both_have_suffix": both_have,
            "suffix_match": suffix_match,
        }
    
    def extract_composite_features(self, name1: str, name2: str,
                                   addr1: str, addr2: str) -> Dict[str, float]:
        """
        Extract composite/interaction features (Phase 2).
        
        Features (7 total):
        - harmonic_mean: 2*name_sim*addr_sim / (name_sim + addr_sim)
        - product: name_sim * addr_sim (normalized)
        - minimum: min(name_sim, addr_sim)
        - maximum: max(name_sim, addr_sim)
        - abs_diff: abs(name_sim - addr_sim)
        - high_both: 1.0 if both > 0.75
        - composite_score: weighted combination
        
        Args:
            name1, name2: Normalized names
            addr1, addr2: Normalized addresses
        
        Returns:
            Dict with 7 composite features
        """
        # Get base similarities
        if RAPIDFUZZ_AVAILABLE:
            name_sim = fuzz.ratio(name1, name2) / 100.0
            addr_sim = fuzz.ratio(addr1, addr2) / 100.0
        else:
            name_sim = 0.5
            addr_sim = 0.5
        
        # Harmonic mean (captures "both high" constraint)
        if (name_sim + addr_sim) > 0:
            harmonic = (2 * name_sim * addr_sim) / (name_sim + addr_sim)
        else:
            harmonic = 0.0
        
        # Product (multiplicative interaction)
        product = name_sim * addr_sim
        
        # Min/Max (weakest-link and strongest-link)
        minimum = min(name_sim, addr_sim)
        maximum = max(name_sim, addr_sim)
        
        # Absolute difference (flags discrepancies)
        abs_diff = abs(name_sim - addr_sim)
        
        # Both high indicator
        high_both = 1.0 if (name_sim > 0.75 and addr_sim > 0.75) else 0.0
        
        # Weighted composite
        composite_score = 0.5 * name_sim + 0.5 * addr_sim
        
        return {
            "harmonic_mean": harmonic,
            "product": product,
            "minimum": minimum,
            "maximum": maximum,
            "abs_diff": abs_diff,
            "high_both": high_both,
            "composite_score": composite_score,
        }
    
    def extract_postal_hierarchical_features(self, addr1: str, addr2: str) -> Dict[str, float]:
        """
        Extract hierarchical postal code features (Phase 2).
        
        Features (4 total):
        - postal_exact: 1.0 if full postal codes match
        - postal_prefix_2: 1.0 if first 2 digits match
        - postal_prefix_3: 1.0 if first 3 digits match
        - postal_distance: Numeric distance between codes
        
        Args:
            addr1, addr2: Normalized addresses
        
        Returns:
            Dict with 4 postal features
        """
        from .normalize import extract_postal_code
        
        postal1 = extract_postal_code(addr1)
        postal2 = extract_postal_code(addr2)
        
        # Exact match
        postal_exact = 1.0 if (postal1 and postal2 and postal1 == postal2) else 0.0
        
        # Prefix matches
        postal_prefix_2 = 1.0 if (postal1 and postal2 and postal1[:2] == postal2[:2]) else 0.0
        postal_prefix_3 = 1.0 if (postal1 and postal2 and postal1[:3] == postal2[:3]) else 0.0
        
        # Distance (for numeric postal codes)
        postal_distance = 0.0
        if postal1 and postal2:
            try:
                dist = abs(int(postal1) - int(postal2))
                postal_distance = 1.0 / (1.0 + dist / 100.0)  # Normalized distance
            except ValueError:
                postal_distance = 0.0
        
        return {
            "postal_exact": postal_exact,
            "postal_prefix_2": postal_prefix_2,
            "postal_prefix_3": postal_prefix_3,
            "postal_distance": postal_distance,
        }
    
    def extract_landmark_features(self, addr1: str, addr2: str) -> Dict[str, float]:
        """
        Extract landmark/domain features (Phase 2).
        
        Features (2 total):
        - landmark_overlap: Shared landmark references
        - domain_match: Same geographic domain/locality
        
        Args:
            addr1, addr2: Normalized addresses
        
        Returns:
            Dict with 2 landmark features
        """
        # Extract landmark keywords
        landmarks = ['near', 'beside', 'opp', 'opposite', 'landmark', 'building', 'gate', 'tower']
        
        addr1_lower = addr1.lower()
        addr2_lower = addr2.lower()
        
        # Check if both mention landmarks
        landmarks1 = [l for l in landmarks if l in addr1_lower]
        landmarks2 = [l for l in landmarks if l in addr2_lower]
        
        landmark_overlap = 1.0 if (landmarks1 and landmarks2 and set(landmarks1) & set(landmarks2)) else 0.0
        
        # Extract domain (city/region patterns)
        domain_keywords = ['street', 'road', 'avenue', 'boulevard', 'lane', 'highway']
        domains1 = [d for d in domain_keywords if d in addr1_lower]
        domains2 = [d for d in domain_keywords if d in addr2_lower]
        
        domain_match = 1.0 if (domains1 and domains2 and set(domains1) & set(domains2)) else 0.0
        
        return {
            "landmark_overlap": landmark_overlap,
            "domain_match": domain_match,
        }
    
    def extract_multichannel_features(self, s1_id: str, s2_s3_id: str,
                                     channel_ranks: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        """
        Extract multi-channel retrieval indicator features (Phase 2, Task 4).
        
        Features (19 total):
        Per-channel indicators (8):
        - ch1_name_tokens, ch2_addr_tokens, ch3_street_number, ch4_postal: 1.0 if candidate retrieved by this channel
        - ch1_rank, ch2_rank, ch3_rank, ch4_rank: Normalized rank (0-1) per channel
        
        Best-rank features (4):
        - best_channel_rank: Lowest rank across channels
        - rank_agreement: Variance in ranks (lower = more agreement)
        - reciprocal_rank: 1 / (1 + best_rank)
        - channel_count: How many channels retrieved this pair
        
        Agreement features (7):
        - channels_agree: 1.0 if all channels agree on presence
        - name_addr_agreement: 1.0 if name and addr channels agree
        - top_channel: Rank of top-performing channel
        - channel_diversity: Entropy over channel rankings
        - cross_channel_rank: Average rank across channels
        - avg_channel_rank: Mean of available ranks
        - channel_confidence: Combined confidence from channel agreement
        
        Args:
            s1_id: S1 entity ID
            s2_s3_id: S2/S3 entity ID
            channel_ranks: Optional dict mapping channel name -> rank
        
        Returns:
            Dict with 19 multi-channel features
        """
        # Default: no channel information
        if channel_ranks is None:
            channel_ranks = {}
        
        # Per-channel indicators
        ch1_retrieved = 1.0 if "name_tokens" in channel_ranks else 0.0
        ch2_retrieved = 1.0 if "addr_tokens" in channel_ranks else 0.0
        ch3_retrieved = 1.0 if "street_number" in channel_ranks else 0.0
        ch4_retrieved = 1.0 if "postal" in channel_ranks else 0.0
        
        # Per-channel ranks (normalized to [0, 1])
        def normalize_rank(rank, max_rank=100):
            if rank is None or rank < 0:
                return 0.0
            return 1.0 / (1.0 + rank / max_rank)
        
        ch1_rank = normalize_rank(channel_ranks.get("name_tokens"))
        ch2_rank = normalize_rank(channel_ranks.get("addr_tokens"))
        ch3_rank = normalize_rank(channel_ranks.get("street_number"))
        ch4_rank = normalize_rank(channel_ranks.get("postal"))
        
        # Best-rank features
        ranks_list = [r for r in channel_ranks.values() if r is not None and r >= 0]
        if ranks_list:
            best_rank = min(ranks_list)
            best_channel_rank = normalize_rank(best_rank)
            
            # Rank variance (lower = more agreement)
            if len(ranks_list) > 1:
                rank_variance = np.var(ranks_list)
                rank_agreement = 1.0 / (1.0 + rank_variance / 100.0)
            else:
                rank_agreement = 1.0
            
            # Reciprocal rank
            reciprocal_rank = 1.0 / (1.0 + best_rank)
            
            # Channel count
            channel_count = len(ranks_list) / 4.0  # Normalize by max channels
        else:
            best_channel_rank = 0.0
            rank_agreement = 0.0
            reciprocal_rank = 0.0
            channel_count = 0.0
        
        # Agreement features
        channels_agree = 1.0 if len(set(ranks_list)) <= 1 else 0.0
        name_addr_agreement = 1.0 if ("name_tokens" in channel_ranks and "addr_tokens" in channel_ranks) else 0.0
        
        # Top channel
        if channel_ranks:
            top_channel_rank = min(channel_ranks.values())
            top_channel = normalize_rank(top_channel_rank)
        else:
            top_channel = 0.0
        
        # Channel diversity (entropy)
        if ranks_list and len(ranks_list) > 1:
            # Normalize ranks to probabilities
            ranks_array = np.array(ranks_list)
            min_rank = ranks_array.min()
            max_rank = ranks_array.max()
            if max_rank > min_rank:
                probs = (max_rank - ranks_array) / (max_rank - min_rank)
                probs = probs / probs.sum()
                # Entropy
                channel_diversity = -np.sum(probs * np.log(probs + 1e-10))
                # Normalize to [0, 1]
                channel_diversity = channel_diversity / np.log(len(probs) + 1)
            else:
                channel_diversity = 0.0
        else:
            channel_diversity = 0.0
        
        # Cross-channel rank
        cross_channel_rank = np.mean(ranks_list) / 100.0 if ranks_list else 0.0
        avg_channel_rank = np.mean([r for r in [ch1_rank, ch2_rank, ch3_rank, ch4_rank] if r > 0]) if any([ch1_rank, ch2_rank, ch3_rank, ch4_rank]) else 0.0
        
        # Channel confidence
        channel_confidence = (ch1_retrieved + ch2_retrieved + ch3_retrieved + ch4_retrieved) / 4.0
        
        return {
            # Per-channel indicators (8)
            "ch1_name_tokens": ch1_retrieved,
            "ch2_addr_tokens": ch2_retrieved,
            "ch3_street_number": ch3_retrieved,
            "ch4_postal": ch4_retrieved,
            "ch1_rank": ch1_rank,
            "ch2_rank": ch2_rank,
            "ch3_rank": ch3_rank,
            "ch4_rank": ch4_rank,
            
            # Best-rank features (4)
            "best_channel_rank": best_channel_rank,
            "rank_agreement": rank_agreement,
            "reciprocal_rank": reciprocal_rank,
            "channel_count": channel_count,
            
            # Agreement features (7)
            "channels_agree": channels_agree,
            "name_addr_agreement": name_addr_agreement,
            "top_channel": top_channel,
            "channel_diversity": channel_diversity,
            "cross_channel_rank": cross_channel_rank,
            "avg_channel_rank": avg_channel_rank,
            "channel_confidence": channel_confidence,
        }
    
    def extract_features_for_pair(self, s1_record: Dict, s2_s3_record: Dict,
                                 phase: int = 2, channel_ranks: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        """
        Extract all features for a candidate pair.
        
        Phase 1: 17 core features
        Phase 2: 55 advanced features (all)
        Phase 2 includes multi-channel meta-features if channel_ranks provided
        
        Args:
            s1_record: Source 1 record dict
            s2_s3_record: Source 2/3 record dict
            phase: 1 (17 features) or 2 (55 features)
            channel_ranks: Optional dict mapping channel name -> rank
        
        Returns:
            Dict mapping feature_name -> value
        """
        # Normalize both sides
        s1_name = normalize_name(s1_record.get('business_name', ''))
        s1_addr = normalize_address(s1_record.get('business_address', ''))
        
        s2_s3_name = normalize_name(s2_s3_record.get('business_name', ''))
        s2_s3_addr = normalize_address(s2_s3_record.get('business_address', ''))
        
        # Extract all feature groups
        features = {}
        
        # Phase 1 features (always included)
        features.update(self.extract_name_similarity_features(s1_name, s2_s3_name))
        features.update(self.extract_address_similarity_features(s1_addr, s2_s3_addr))
        features['number_overlap'] = self.extract_number_overlap_feature(s1_addr, s2_s3_addr)
        features.update(self.extract_token_level_features(s1_name, s2_s3_name, s1_addr, s2_s3_addr))
        features.update(self.extract_exact_match_features(s1_name, s2_s3_name, s1_addr, s2_s3_addr))
        features.update(self.extract_structural_features(s1_name, s2_s3_name, s1_addr, s2_s3_addr))
        
        # Phase 5 reference integration: 3-gram Jaccard (typo tolerance)
        features.update(self.extract_name_ngram_jaccard(s1_name, s2_s3_name))
        features.update(self.extract_address_ngram_jaccard(s1_addr, s2_s3_addr))
        
        # Phase 2 features (if phase >= 2)
        if phase >= 2:
            features.update(self.extract_legal_suffix_features(s1_name, s2_s3_name))
            features.update(self.extract_composite_features(s1_name, s2_s3_name, s1_addr, s2_s3_addr))
            features.update(self.extract_postal_hierarchical_features(s1_addr, s2_s3_addr))
            features.update(self.extract_landmark_features(s1_addr, s2_s3_addr))
            
            # Multi-channel meta-features (Task 4)
            s1_id = s1_record.get('entity_id', 'unknown')
            s2_s3_id = s2_s3_record.get('entity_id', 'unknown')
            features.update(self.extract_multichannel_features(s1_id, s2_s3_id, channel_ranks))
        
        return features
    
    def extract_features_batch(self, candidate_pairs: List[Tuple[str, str]],
                              s1_records: Dict[str, Dict],
                              s2_s3_records: Dict[str, Dict],
                              phase: int = 2,
                              channel_ranks_dict: Optional[Dict[Tuple[str, str], Dict[str, int]]] = None) -> pd.DataFrame:
        """
        Extract features for batch of candidate pairs.
        
        Args:
            candidate_pairs: List of (s1_id, s2_s3_id) tuples
            s1_records: S1 record dicts
            s2_s3_records: S2/S3 record dicts
            phase: 1 (17 features) or 2 (55 features)
            channel_ranks_dict: Optional dict mapping (s1_id, s2_s3_id) -> channel_ranks dict
        
        Returns:
            Pandas DataFrame with features (one row per pair)
        """
        features_list = []
        
        for s1_id, s2_s3_id in candidate_pairs:
            if s1_id not in s1_records or s2_s3_id not in s2_s3_records:
                continue
            
            # Get channel ranks if available
            channel_ranks = None
            if channel_ranks_dict and (s1_id, s2_s3_id) in channel_ranks_dict:
                channel_ranks = channel_ranks_dict[(s1_id, s2_s3_id)]
            
            features = self.extract_features_for_pair(
                s1_records[s1_id],
                s2_s3_records[s2_s3_id],
                phase=phase,
                channel_ranks=channel_ranks
            )
            
            # Add IDs
            features['s1_id'] = s1_id
            features['s2_s3_id'] = s2_s3_id
            
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        
        expected_features = 55 if phase >= 2 else 17
        if self.verbose:
            print(f"   Extracted {len(df):,} feature vectors ({len(df.columns)-2} features)")
        
        return df


def extract_features_from_records(candidates_by_s1: Dict[str, List[Tuple[str, float]]],
                                 s1_records: Dict[str, Dict],
                                 s23_records: Dict[str, Dict],
                                 phase: int = 2,
                                 channel_ranks_dict: Optional[Dict[Tuple[str, str], Dict[str, int]]] = None,
                                 verbose: bool = True) -> pd.DataFrame:
    """Feature extraction from already-materialised record dicts.

    ``extract_features_from_candidates`` takes DataFrames and rebuilds the
    record dicts with ``iterrows`` on every call. That is wasteful when
    scoring in chunks, because the same source records are re-converted for
    every chunk. This variant takes the dicts directly so the caller can
    normalise each record once and reuse it across chunks.
    """
    pairs = []
    for s1_id, candidates in candidates_by_s1.items():
        s1_rec = s1_records.get(s1_id)
        if s1_rec is None:
            continue
        for s2_s3_id, score in candidates:
            if s2_s3_id in s23_records:
                pairs.append((s1_id, s2_s3_id))

    if not pairs:
        return pd.DataFrame(columns=list(FEATURE_NAMES) + ["s1_id", "s2_s3_id"])

    extractor = FeatureExtractor(verbose=verbose)
    features_df = extractor.extract_features_batch(
        pairs, s1_records, s23_records,
        phase=phase, channel_ranks_dict=channel_ranks_dict,
    )
    return features_df


def extract_features_from_candidates(candidates_by_s1: Dict[str, List[Tuple[str, float]]],
                                     s1_df: pd.DataFrame, s2_df: pd.DataFrame,
                                     s3_df: pd.DataFrame, phase: int = 2,
                                     channel_ranks_dict: Optional[Dict[Tuple[str, str], Dict[str, int]]] = None,
                                     verbose: bool = True) -> pd.DataFrame:
    """
    Convenience function to extract features from blocking output.
    
    Args:
        candidates_by_s1: Output from blocker
        s1_df: Source 1 dataframe
        s2_df: Source 2 dataframe
        s3_df: Source 3 dataframe
        phase: 1 (17 features) or 2 (55 features)
        channel_ranks_dict: Optional dict mapping (s1_id, s2_s3_id) -> channel_ranks
        verbose: Print progress
    
    Returns:
        Pandas DataFrame with extracted features
    """
    if verbose:
        print(f"\n🎯 Extracting features (Phase {phase})...")
    
    # Convert to record dicts
    s1_records = {row['entity_id']: row.to_dict() for _, row in s1_df.iterrows()}
    s2_records = {row['entity_id']: row.to_dict() for _, row in s2_df.iterrows()}
    s3_records = {row['entity_id']: row.to_dict() for _, row in s3_df.iterrows()}
    s2_s3_records = {**s2_records, **s3_records}
    
    # Build candidate pairs list
    candidate_pairs = []
    for s1_id, candidates in candidates_by_s1.items():
        for s2_s3_id, score in candidates:
            candidate_pairs.append((s1_id, s2_s3_id))
    
    # Extract features
    extractor = FeatureExtractor(verbose=verbose)
    features_df = extractor.extract_features_batch(candidate_pairs, s1_records, s2_s3_records, 
                                                   phase=phase, channel_ranks_dict=channel_ranks_dict)
    
    return features_df


if __name__ == "__main__":
    # Test feature extraction
    print("=" * 80)
    print("Testing FeatureExtractor")
    print("=" * 80)
    
    extractor = FeatureExtractor(verbose=True)
    
    # Test pair
    s1_record = {
        'business_name': 'ACME Corporation',
        'business_address': '456 Oak Street, Suite 789, New York, NY 10001',
    }
    
    s2_record = {
        'business_name': 'ACME Corp',
        'business_address': '456 Oak Lane, Apt 789, New York, NY 10001',
    }
    
    features = extractor.extract_features_for_pair(s1_record, s2_record)
    
    print("\nExtracted features:")
    for feat_name, feat_value in sorted(features.items()):
        print(f"  {feat_name:30s}: {feat_value:.4f}")
    
    print(f"\n✅ Feature extraction test completed!")


class RecView(Mapping):
    """Expose ``(name, address)`` tuples as record dicts, one at a time.

    Pools are held as tuples rather than record dicts because a 4-key dict
    per record costs ~400 bytes of headers alone, which at the test pool's
    9.97M records is ~4 GB before any string data. The extractor wants
    dicts, so this adapts on access rather than copying the pool into a
    second dict-of-dicts.
    """

    __slots__ = ("_src",)

    def __init__(self, src):
        self._src = src

    def __getitem__(self, key):
        t = self._src[key]
        return {"business_name": t[0] or "", "business_address": t[1] or ""}

    def __iter__(self):
        return iter(self._src)

    def __len__(self):
        return len(self._src)


def fixed_chunks(mapping, size):
    """Split a materialised candidate dict into ``size``-sized pieces."""
    items = list(mapping.items())
    for i in range(0, len(items), size):
        block = dict(items[i:i + size])
        yield list(block), block
