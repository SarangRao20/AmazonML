"""Create a representative stratified validation sample for fast local development."""

import os
import pandas as pd

def create_sample(
    data_dir: str = "dataset/train",
    out_dir: str = "dataset/val_sample",
    n_s1: int = 50000,
    random_state: int = 42
):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading S1 from {data_dir}/train_source1.tsv...")
    s1 = pd.read_csv(f"{data_dir}/train_source1.tsv", sep="\t", dtype=str).fillna("")
    gt = pd.read_csv(f"{data_dir}/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    
    # Merge to stratify by country and singleton status
    gt_map = dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"]))
    s1["matched_entity_ids"] = s1["entity_id"].map(gt_map).fillna("")
    s1["is_singleton"] = s1["matched_entity_ids"] == ""
    
    # Stratified sample
    groups = []
    for (country, is_sing), group in s1.groupby(["country", "is_singleton"]):
        n_group = max(1, int(len(group) * (n_s1 / len(s1))))
        groups.append(group.sample(n=n_group, random_state=random_state))
    sample_s1 = pd.concat(groups).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    print(f"Sampled {len(sample_s1)} S1 entities.")
    print("Country distribution:\n", sample_s1["country"].value_counts().to_dict())
    print("Singleton distribution:\n", sample_s1["is_singleton"].value_counts().to_dict())
    
    # Collect all true matching S2 and S3 IDs
    matching_s2 = set()
    matching_s3 = set()
    for m in sample_s1["matched_entity_ids"]:
        if not m:
            continue
        for x in m.split(","):
            if x.startswith("S2-"):
                matching_s2.add(x)
            elif x.startswith("S3-"):
                matching_s3.add(x)
                
    print(f"Found {len(matching_s2)} true S2 matches, {len(matching_s3)} true S3 matches.")
    
    # Save S1 and Ground Truth
    sample_s1[["entity_id", "business_name", "business_address", "country"]].to_csv(
        f"{out_dir}/sample_source1.tsv", sep="\t", index=False
    )
    sample_s1[["entity_id", "matched_entity_ids"]].rename(
        columns={"entity_id": "source1_entity_id"}
    ).to_csv(
        f"{out_dir}/sample_ground_truth.tsv", sep="\t", index=False
    )
    
    # Now extract S2 (matching + random negative distractors)
    print("Loading S2...")
    s2 = pd.read_csv(f"{data_dir}/train_source2.tsv", sep="\t", dtype=str).fillna("")
    s2_match_mask = s2["entity_id"].isin(matching_s2)
    s2_matches = s2[s2_match_mask]
    s2_distractors = s2[~s2_match_mask].sample(n=min(len(s2) - len(s2_matches), len(matching_s2) * 2), random_state=random_state)
    sample_s2 = pd.concat([s2_matches, s2_distractors]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    sample_s2.to_csv(f"{out_dir}/sample_source2.tsv", sep="\t", index=False)
    print(f"Saved {len(sample_s2)} S2 records.")
    del s2, s2_matches, s2_distractors
    
    # Now extract S3 (matching + random negative distractors)
    print("Loading S3...")
    s3 = pd.read_csv(f"{data_dir}/train_source3.tsv", sep="\t", dtype=str).fillna("")
    s3_match_mask = s3["entity_id"].isin(matching_s3)
    s3_matches = s3[s3_match_mask]
    s3_distractors = s3[~s3_match_mask].sample(n=min(len(s3) - len(s3_matches), len(matching_s3) * 2), random_state=random_state)
    sample_s3 = pd.concat([s3_matches, s3_distractors]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    sample_s3.to_csv(f"{out_dir}/sample_source3.tsv", sep="\t", index=False)
    print(f"Saved {len(sample_s3)} S3 records.")
    
    print(f"Validation sample dataset ready at {out_dir}!")

if __name__ == "__main__":
    create_sample()
