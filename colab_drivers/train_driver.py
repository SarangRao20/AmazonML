"""Full-train driver (stratified sample, full per-country pools).

Stages: sample -> block (full pool) -> featurize -> GroupKFold tri-ensemble
-> vectorized threshold search. Every artifact checkpoints to Drive.
Run:  SAMPLE_N=150000 nohup python3 train_driver.py > ckpt/train.log 2>&1 &
"""
import os, sys, json, gc, time
from pathlib import Path

os.environ["AMZ_ML_ROOT"] = "/content/work"
sys.path.insert(0, "/content/work")
t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:8.1f}s] {msg}", flush=True)

import polars as pl
import pandas as pd
import numpy as np

from code.business_entity_resolution.src import model as model_mod
from code.business_entity_resolution.src.blocking import MultiChannelBlocker
from code.business_entity_resolution.src.features import extract_features_from_candidates
from code.business_entity_resolution.src.model import TriEnsembleModel
from code.business_entity_resolution.src.threshold_optimizer_fast import VectorizedThresholdOptimizer

DRIVE = Path("/content/drive/MyDrive/AmazonML")
DATA = DRIVE / "dataset"
CKPT = DRIVE / "checkpoints"
(CKPT / "models").mkdir(exist_ok=True, parents=True)
SAMPLE_N = int(os.environ.get("SAMPLE_N", "150000"))
N_FOLDS = int(os.environ.get("N_FOLDS", "3"))
SEED = 42

def load_tsv(path):
    return pl.read_csv(path, separator="\t").to_pandas()

# ---------- 1. load train + gt ----------
log("loading train files")
s1 = load_tsv(DATA / "train" / "train_source1.tsv")
s2 = load_tsv(DATA / "train" / "train_source2.tsv")
s3 = load_tsv(DATA / "train" / "train_source3.tsv")
gt = load_tsv(DATA / "train" / "train_ground_truth.tsv")
log(f"s1={len(s1):,} s2={len(s2):,} s3={len(s3):,} gt={len(gt):,}")
gt_dict = {}
for sid, m in zip(gt["source1_entity_id"].values, gt["matched_entity_ids"].fillna("").values):
    gt_dict[sid] = set(m.split(",")) if m else set()

# ---------- 2. stratified sample by country x singleton ----------
samp_ckpt = CKPT / f"sample_s1_{SAMPLE_N}.txt"
if samp_ckpt.exists():
    sample_ids = samp_ckpt.read_text().split()
    log(f"resuming sample: {len(sample_ids):,} ids from ckpt")
else:
    s1["_singleton"] = s1["entity_id"].map(lambda i: len(gt_dict.get(i, set())) == 0)
    rng = np.random.RandomState(SEED)
    parts = []
    for (c, sg), grp in s1.groupby(["country", "_singleton"]):
        n = max(1, round(len(grp) * SAMPLE_N / len(s1)))
        parts.append(grp.sample(n=min(n, len(grp)), random_state=SEED))
    samp = pd.concat(parts).sample(frac=1.0, random_state=SEED).head(SAMPLE_N)
    sample_ids = samp["entity_id"].tolist()
    samp_ckpt.write_text("\n".join(sample_ids))
    log(f"sampled {len(sample_ids):,} S1 ({SAMPLE_N} target)")
sample_set = set(sample_ids)
sample_gt = {i: gt_dict.get(i, set()) for i in sample_ids}
s1_samp = s1[s1["entity_id"].isin(sample_set)].copy()
del gt
gc.collect()

# ---------- 3+4. per-country block + featurize ----------
feat_parts = []
for country in ["India", "US"]:
    c_ckpt = CKPT / f"train_feat_{country}_{SAMPLE_N}.parquet"
    if c_ckpt.exists():
        feat_parts.append(pd.read_parquet(c_ckpt))
        log(f"{country}: features from ckpt ({len(feat_parts[-1]):,} pairs)")
        continue
    s1c = s1_samp[s1_samp["country"] == country]
    pool = pd.concat([s2[s2["country"] == country], s3[s3["country"] == country]])
    log(f"{country}: blocking {len(s1c):,} S1 vs pool {len(pool):,}")
    s1_rec = {r["entity_id"]: r for r in s1c.to_dict("records")}
    pool_rec = {r["entity_id"]: r for r in pool.to_dict("records")}
    del pool
    gc.collect()
    blocker = MultiChannelBlocker(verbose=True)
    cands = blocker.generate_all_candidates(s1_rec, pool_rec)
    rec = blocker.measure_recall(cands, {k: sample_gt[k] for k in s1_rec})
    log(f"{country}: recall={rec*100:.2f}%")
    del s1_rec, pool_rec
    gc.collect()
    s1c_df = s1c.drop(columns=["_singleton"])
    pool_df = pd.concat([s2[s2["country"] == country], s3[s3["country"] == country]])
    feats = extract_features_from_candidates(cands, s1c_df,
        pool_df[pool_df["entity_id"].str.startswith("S2-")],
        pool_df[pool_df["entity_id"].str.startswith("S3-")],
        phase=2, verbose=True)
    del cands, pool_df, s1c_df
    gc.collect()
    lab = np.array([1 if c in sample_gt.get(s, set()) else 0
                    for s, c in zip(feats["s1_id"].values, feats["s2_s3_id"].values)],
                   dtype=np.int8)
    feats["label"] = lab
    log(f"{country}: pairs={len(feats):,} pos={lab.sum():,}")
    feats.to_parquet(c_ckpt, index=False)
    feat_parts.append(feats)
del s1, s2, s3, s1_samp
gc.collect()

features_df = pd.concat(feat_parts, ignore_index=True)
del feat_parts
gc.collect()
log(f"all features: {len(features_df):,} pairs")

# ---------- 5. train ----------
model_mod.N_SPLITS_CV = N_FOLDS
groups = features_df["s1_id"].map(
    {v: i for i, v in enumerate(features_df["s1_id"].unique())}).values
X = features_df.drop(columns=["label"])
y = features_df["label"].values
log(f"training {N_FOLDS}-fold tri-ensemble on {len(X):,} pairs")
ens = TriEnsembleModel(verbose=True)
trained, oof = ens.train_groupkfold(X, y, groups)
ens.save_models(trained, CKPT / "models")

# ---------- 6. thresholds on OOF ----------
oof_sorted = oof.set_index("row_index").reindex(features_df.index)
probs = oof_sorted["blend_prob"].to_numpy(dtype=np.float64)
s1_arr = features_df["s1_id"].to_numpy()
cand_arr = features_df["s2_s3_id"].to_numpy()
order = np.argsort(s1_arr, kind="stable")
ss, cs, ps = s1_arr[order], cand_arr[order], probs[order]
uniq, starts = np.unique(ss, return_index=True)
counts = np.diff(np.append(starts, len(ss)))
prob_by_s1 = {}
for sid, st, cn in zip(uniq, starts, counts):
    sl = slice(st, st + cn)
    prob_by_s1[sid] = list(zip(cs[sl].tolist(), ps[sl].tolist()))
opt = VectorizedThresholdOptimizer(beta=0.5)
st_tau, mg_tau, f, top = opt.grid_search(prob_by_s1, sample_gt)
log(f"BEST score={st_tau} margin={mg_tau} macroF={f*100:.2f}%")
(CKPT / "thresholds.json").write_text(json.dumps(
    {"score_tau": st_tau, "margin_tau": mg_tau, "macro_f": f,
     "sample_n": SAMPLE_N, "folds": N_FOLDS}))
log("DONE")
