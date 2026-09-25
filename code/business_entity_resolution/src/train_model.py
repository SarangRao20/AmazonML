"""Train LightGBM pairwise matching classifier and optimize threshold for Macro F_0.5."""

import os
import time
from collections import defaultdict
import numpy as np
import pandas as pd
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
from .features import extract_pair_features, FEATURE_NAMES
from .metrics import evaluate_macro_f05


def prepare_training_data(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    candidates: dict,
    gt_map: dict
):
    """Extract features and labels for all candidate pairs."""
    print("Pre-processing records for feature extraction...")
    # Precompute S1 preprocessed attributes
    s1_dict = {}
    for _, row in s1_df.iterrows():
        s1_id = row["entity_id"]
        norm_name = normalize_text(row["business_name"])
        core_name = get_core_name(norm_name)
        norm_addr = normalize_text(row["business_address"])
        s1_dict[s1_id] = (
            norm_name,
            core_name,
            norm_addr,
            get_tokens(core_name, 3),
            get_tokens(norm_addr, 3),
            get_numbers(norm_addr),
            get_char_ngrams(core_name, 3)
        )

    # Precompute Pool preprocessed attributes
    pool_dict = {}
    for _, row in pool_df.iterrows():
        p_id = row["entity_id"]
        norm_name = normalize_text(row["business_name"])
        core_name = get_core_name(norm_name)
        norm_addr = normalize_text(row["business_address"])
        pool_dict[p_id] = (
            norm_name,
            core_name,
            norm_addr,
            get_tokens(core_name, 3),
            get_tokens(norm_addr, 3),
            get_numbers(norm_addr),
            get_char_ngrams(core_name, 3)
        )

    print(f"Extracting features for candidate pairs across {len(candidates):,} S1 entities...")
    X_rows = []
    y_rows = []
    pair_meta = [] # (s1_id, cand_id)

    for s1_id, cands in candidates.items():
        s1_attrs = s1_dict[s1_id]
        true_set = gt_map.get(s1_id, set())

        for rank, (cand_id, b_score) in enumerate(cands, start=1):
            if cand_id not in pool_dict:
                continue
            cand_attrs = pool_dict[cand_id]
            feats = extract_pair_features(
                s1_attrs[0], s1_attrs[1], s1_attrs[2], s1_attrs[3], s1_attrs[4], s1_attrs[5], s1_attrs[6],
                cand_attrs[0], cand_attrs[1], cand_attrs[2], cand_attrs[3], cand_attrs[4], cand_attrs[5], cand_attrs[6],
                b_score, rank
            )
            label = 1 if cand_id in true_set else 0
            X_rows.append(feats)
            y_rows.append(label)
            pair_meta.append((s1_id, cand_id))

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32), pair_meta


def train_and_evaluate(
    val_sample_dir: str = "dataset/val_sample",
    model_save_path: str = "models/lgbm_matcher.txt"
):
    print("Loading validation sample...")
    s1_df = pd.read_csv(f"{val_sample_dir}/sample_source1.tsv", sep="\t", dtype=str).fillna("")
    s2_df = pd.read_csv(f"{val_sample_dir}/sample_source2.tsv", sep="\t", dtype=str).fillna("")
    s3_df = pd.read_csv(f"{val_sample_dir}/sample_source3.tsv", sep="\t", dtype=str).fillna("")
    gt_df = pd.read_csv(f"{val_sample_dir}/sample_ground_truth.tsv", sep="\t", dtype=str).fillna("")

    gt_map = {
        row["source1_entity_id"]: set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
        for _, row in gt_df.iterrows()
    }

    # Split S1 into Train (70%) and Val (30%)
    np.random.seed(42)
    s1_ids = s1_df["entity_id"].values
    shuffled_idx = np.random.permutation(len(s1_ids))
    n_train = int(len(s1_ids) * 0.70)

    train_s1_ids = set(s1_ids[shuffled_idx[:n_train]])
    val_s1_ids = set(s1_ids[shuffled_idx[n_train:]])

    train_s1 = s1_df[s1_df["entity_id"].isin(train_s1_ids)].reset_index(drop=True)
    val_s1 = s1_df[s1_df["entity_id"].isin(val_s1_ids)].reset_index(drop=True)

    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)

    print(f"Train S1: {len(train_s1):,}, Val S1: {len(val_s1):,}, Pool: {len(pool_df):,}")

    # Run blocking
    blocker = MultiPassBlocker(top_k=35)
    print("\n--- Running Blocking for Train Split ---")
    train_cands = blocker.block_all(train_s1, s2_df, s3_df)

    print("\n--- Running Blocking for Val Split ---")
    val_cands = blocker.block_all(val_s1, s2_df, s3_df)

    # Feature extraction
    X_train, y_train, train_meta = prepare_training_data(train_s1, pool_df, train_cands, gt_map)
    X_val, y_val, val_meta = prepare_training_data(val_s1, pool_df, val_cands, gt_map)

    print(f"Train dataset: {X_train.shape[0]:,} pairs (Positives: {y_train.sum():,}, {y_train.mean()*100:.2f}%)")
    print(f"Val dataset:   {X_val.shape[0]:,} pairs (Positives: {y_val.sum():,}, {y_val.mean()*100:.2f}%)")

    # Train LightGBM
    print("\n--- Training LightGBM Classifier ---")
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.08,
        "feature_fraction": 0.85,
        "n_jobs": 8,
        "verbose": -1,
        "seed": 42
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(50)]
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Feature importance
    print("\nFeature Importances:")
    importances = model.feature_importance(importance_type="gain")
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name:25s}: {imp:10.1f}")

    # Optimize threshold for macro F_0.5 on Val Set
    print("\n--- Optimizing Decision Threshold on Val Set ---")
    val_preds_prob = model.predict(X_val)

    # Group predictions by s1_id
    val_pair_probs = defaultdict(list)
    for (s1_id, cand_id), prob in zip(val_meta, val_preds_prob):
        val_pair_probs[s1_id].append((cand_id, prob))

    best_thresh = 0.5
    best_f05 = -1.0
    best_metrics = None

    thresholds = np.arange(0.35, 0.85, 0.05)
    for thresh in thresholds:
        pred_matches = {}
        for s1_id in val_s1["entity_id"]:
            cands = val_pair_probs.get(s1_id, [])
            matched = {cand_id for cand_id, p in cands if p >= thresh}
            pred_matches[s1_id] = matched

        val_gt = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1["entity_id"]}
        m = evaluate_macro_f05(val_gt, pred_matches)
        print(f"Thresh={thresh:.2f} -> Macro F_0.5: {m['macro_f05']:.4f} | Non-singleton: {m['non_singleton_f05']:.4f} | Singleton Acc: {m['singleton_acc']:.4f} | Micro P: {m['micro_precision']:.4f} | Micro R: {m['micro_recall']:.4f}")

        if m["macro_f05"] > best_f05:
            best_f05 = m["macro_f05"]
            best_thresh = thresh
            best_metrics = m

    print(f"\n==========================================")
    print(f"Best Validation Macro F_0.5: {best_f05:.4f} at threshold = {best_thresh:.2f}")
    print(f"Non-singleton F_0.5:        {best_metrics['non_singleton_f05']:.4f}")
    print(f"Singleton Accuracy:         {best_metrics['singleton_acc']:.4f}")
    print(f"Micro Precision:            {best_metrics['micro_precision']:.4f}")
    print(f"Micro Recall:               {best_metrics['micro_recall']:.4f}")
    print(f"==========================================")


if __name__ == "__main__":
    train_and_evaluate()
