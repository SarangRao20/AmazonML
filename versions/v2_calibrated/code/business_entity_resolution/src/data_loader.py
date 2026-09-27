"""
Data loading and entity-level disjoint train/validation splitting.

Critical Design: Entity-level DISJOINT split prevents data leakage.
When S1 entity appears in train, ALL its candidate pairs (S1-S2, S1-S3) are in train.
When S1 entity is in validation, ALL its pairs are in validation.
This ensures zero entity overlap between splits.
"""

import pandas as pd
import polars as pl
from pathlib import Path
from typing import Tuple, Dict, List, Set
from collections import defaultdict
import numpy as np
from sklearn.model_selection import GroupKFold

from .config import (
    TRAIN_SOURCE1_PATH, TRAIN_SOURCE2_PATH, TRAIN_SOURCE3_PATH,
    TRAIN_GROUND_TRUTH_PATH, VAL_SOURCE1_PATH, VAL_SOURCE2_PATH,
    VAL_SOURCE3_PATH, VAL_GROUND_TRUTH_PATH, USE_VAL_SAMPLE,
    VALIDATION_SIZE, TRAIN_VAL_SPLIT_RATIO, N_SPLITS_CV, RANDOM_SEED,
    VERBOSE
)


class DataLoader:
    """Load and preprocess entity resolution data."""
    
    def __init__(self, use_val_sample: bool = True, verbose: bool = True):
        """
        Initialize data loader.
        
        Args:
            use_val_sample: If True, use val_sample dataset (faster for testing).
                           If False, use full training data.
            verbose: Print progress information.
        """
        self.use_val_sample = use_val_sample
        self.verbose = verbose
        
        # Select paths based on dataset choice
        if use_val_sample:
            self.s1_path = VAL_SOURCE1_PATH
            self.s2_path = VAL_SOURCE2_PATH
            self.s3_path = VAL_SOURCE3_PATH
            self.gt_path = VAL_GROUND_TRUTH_PATH
            if self.verbose:
                print("📂 Using validation sample dataset (val_sample/)")
        else:
            self.s1_path = TRAIN_SOURCE1_PATH
            self.s2_path = TRAIN_SOURCE2_PATH
            self.s3_path = TRAIN_SOURCE3_PATH
            self.gt_path = TRAIN_GROUND_TRUTH_PATH
            if self.verbose:
                print("📂 Using full training dataset")
        
        self.s1_df = None
        self.s2_df = None
        self.s3_df = None
        self.ground_truth = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all data from TSV files.
        
        Returns:
            Tuple of (s1_df, s2_df, s3_df, ground_truth_df)
        """
        if self.verbose:
            print("\n🔄 Loading data...")
        
        # Use Polars for fast scanning, then convert to Pandas for compatibility
        self.s1_df = pl.read_csv(self.s1_path, separator="\t").to_pandas()
        self.s2_df = pl.read_csv(self.s2_path, separator="\t").to_pandas()
        self.s3_df = pl.read_csv(self.s3_path, separator="\t").to_pandas()
        self.ground_truth = pl.read_csv(self.gt_path, separator="\t").to_pandas()
        
        if self.verbose:
            print(f"   ✓ Source 1: {len(self.s1_df):,} records")
            print(f"   ✓ Source 2: {len(self.s2_df):,} records")
            print(f"   ✓ Source 3: {len(self.s3_df):,} records")
            print(f"   ✓ Ground Truth: {len(self.ground_truth):,} labels")
        
        return self.s1_df, self.s2_df, self.s3_df, self.ground_truth
    
    def get_entity_level_split(self) -> Tuple[List[str], List[str]]:
        """
        Create entity-level DISJOINT train/validation split.
        
        CRITICAL: Splits at S1 entity level, not pair level.
        - Get all unique S1 entity IDs
        - Split them into train_s1_ids and val_s1_ids (disjoint)
        - All pairs (S1_i, S2_j), (S1_i, S3_k) for S1_i in train go to train
        - All pairs for S1_i in validation go to validation
        
        Returns:
            Tuple of (train_s1_ids, val_s1_ids) - lists of S1 entity IDs
        """
        if self.s1_df is None:
            self.load_data()
        
        all_s1_ids = self.s1_df['entity_id'].unique().tolist()
        n_train = int(len(all_s1_ids) * TRAIN_VAL_SPLIT_RATIO)
        
        # Shuffle and split
        np.random.seed(RANDOM_SEED)
        shuffled_ids = np.random.permutation(all_s1_ids)
        train_s1_ids = shuffled_ids[:n_train].tolist()
        val_s1_ids = shuffled_ids[n_train:].tolist()
        
        # Verify disjoint
        assert len(set(train_s1_ids) & set(val_s1_ids)) == 0, "Train/val S1 ids not disjoint!"
        
        if self.verbose:
            print(f"\n✅ Entity-level split (DISJOINT):")
            print(f"   Train S1 entities: {len(train_s1_ids):,}")
            print(f"   Val S1 entities: {len(val_s1_ids):,}")
            print(f"   Overlap check: {len(set(train_s1_ids) & set(val_s1_ids))} (should be 0)")
        
        return train_s1_ids, val_s1_ids
    
    def build_ground_truth_dict(self) -> Dict[str, Set[str]]:
        """
        Build dictionary mapping S1 entity ID -> set of matched S2/S3 IDs.
        
        Returns:
            Dict where keys are S1 entity IDs, values are sets of matched entity IDs
        """
        if self.ground_truth is None:
            self.load_data()
        
        gt_dict = {}
        for _, row in self.ground_truth.iterrows():
            s1_id = row['source1_entity_id']
            matched_ids_str = row.get('matched_entity_ids', '')
            
            if pd.isna(matched_ids_str) or matched_ids_str == '':
                matched_ids = set()  # Singleton
            else:
                matched_ids = set(str(matched_ids_str).split(','))
            
            gt_dict[s1_id] = matched_ids
        
        if self.verbose:
            singletons = sum(1 for ids in gt_dict.values() if len(ids) == 0)
            print(f"\n📊 Ground truth stats:")
            print(f"   Total S1 entities: {len(gt_dict):,}")
            print(f"   Singletons: {singletons:,} ({singletons/len(gt_dict)*100:.2f}%)")
            print(f"   Non-singletons: {len(gt_dict) - singletons:,}")
        
        return gt_dict
    
    def get_groupkfold_splits(self, n_splits: int = N_SPLITS_CV) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get GroupKFold splits where groups are S1 entity IDs.
        
        This ensures all pairs of the same S1 entity stay together
        (either all in train fold or all in val fold for that split).
        
        Args:
            n_splits: Number of folds (default 5)
        
        Returns:
            List of (train_indices, val_indices) tuples
        """
        if self.s1_df is None:
            self.load_data()
        
        # Create dummy X array (indices) and groups (S1 entity IDs)
        X = np.arange(len(self.s1_df))
        groups = self.s1_df['entity_id'].values
        
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X, groups=groups))
        
        if self.verbose:
            print(f"\n🔄 GroupKFold splits (n_splits={n_splits}):")
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                print(f"   Fold {fold_idx + 1}: train={len(train_idx):,}, val={len(val_idx):,}")
        
        return splits
    
    def get_sources_dict(self) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """
        Create dictionaries mapping entity IDs to their records for quick lookup.
        
        Returns:
            Tuple of (s2_dict, s3_dict) where each maps entity_id -> record dict
        """
        if self.s2_df is None or self.s3_df is None:
            self.load_data()
        
        s2_dict = {row['entity_id']: row.to_dict() for _, row in self.s2_df.iterrows()}
        s3_dict = {row['entity_id']: row.to_dict() for _, row in self.s3_df.iterrows()}
        
        if self.verbose:
            print(f"\n📍 Source entity lookups created:")
            print(f"   S2 entities indexed: {len(s2_dict):,}")
            print(f"   S3 entities indexed: {len(s3_dict):,}")
        
        return s2_dict, s3_dict
    
    def build_candidate_pairs(self, train_s1_ids: List[str], s2_dict: Dict, s3_dict: Dict,
                              gt_dict: Dict) -> Tuple[List[Tuple], List[int]]:
        """
        Build candidate pairs from train S1 entities + all S2/S3 records.
        
        For prototyping: pair each S1 with ALL S2 and ALL S3 records.
        In production blocking.py will reduce this to ~35 candidates per S1.
        
        Args:
            train_s1_ids: List of S1 entity IDs for training
            s2_dict: Dictionary of S2 records
            s3_dict: Dictionary of S3 records
            gt_dict: Ground truth dictionary (for labels)
        
        Returns:
            Tuple of (candidate_pairs, labels) where:
            - candidate_pairs: List of (s1_id, s2_or_s3_id) tuples
            - labels: List of binary labels (1 = match, 0 = no match)
        """
        if self.verbose:
            print(f"\n🔗 Building candidate pairs...")
        
        candidate_pairs = []
        labels = []
        
        for s1_id in train_s1_ids:
            if s1_id not in gt_dict:
                continue
            
            matched_ids = gt_dict[s1_id]
            
            # Pair with all S2 records
            for s2_id in s2_dict.keys():
                candidate_pairs.append((s1_id, s2_id))
                labels.append(1 if s2_id in matched_ids else 0)
            
            # Pair with all S3 records
            for s3_id in s3_dict.keys():
                candidate_pairs.append((s1_id, s3_id))
                labels.append(1 if s3_id in matched_ids else 0)
        
        if self.verbose:
            pos_count = sum(labels)
            neg_count = len(labels) - pos_count
            print(f"   Total pairs: {len(candidate_pairs):,}")
            print(f"   Positive (matches): {pos_count:,} ({pos_count/len(labels)*100:.2f}%)")
            print(f"   Negative (non-matches): {neg_count:,} ({neg_count/len(labels)*100:.2f}%)")
            print(f"   Class imbalance ratio: 1:{neg_count/pos_count:.1f}")
        
        return candidate_pairs, labels


def load_train_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quick convenience function to load training data."""
    loader = DataLoader(use_val_sample=USE_VAL_SAMPLE, verbose=VERBOSE)
    return loader.load_data()


if __name__ == "__main__":
    # Test the data loader
    print("=" * 80)
    print("Testing DataLoader")
    print("=" * 80)
    
    loader = DataLoader(use_val_sample=True, verbose=True)
    s1, s2, s3, gt = loader.load_data()
    
    # Get splits
    train_ids, val_ids = loader.get_entity_level_split()
    
    # Build ground truth dict
    gt_dict = loader.build_ground_truth_dict()
    
    # Get GroupKFold splits
    splits = loader.get_groupkfold_splits(n_splits=5)
    
    # Build source dicts and candidate pairs
    s2_dict, s3_dict = loader.get_sources_dict()
    pairs, labels = loader.build_candidate_pairs(train_ids, s2_dict, s3_dict, gt_dict)
    
    print("\n✅ DataLoader test completed successfully!")
