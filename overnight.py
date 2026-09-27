"""Overnight local driver: sampled-pool train + chunked full test inference.

Threaded blocking/feature stages (8 workers), 3-fold tri-ensemble.
Checkpoints in ./checkpoints overnight-safe (resume by design).
Run:  nohup ~/code/ML_DL/.venv/bin/python overnight.py > checkpoints/overnight.log 2>&1 &
"""
import os, sys, json, gc, time, faulthandler, resource
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

faulthandler.enable()  # ponytail: traceback on segfault, silent deaths need a witness

ROOT = Path(__file__).resolve().parent
os.environ["AMZ_ML_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT))
t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:8.1f}s] {msg}", flush=True)

import polars as pl
import pandas as pd
import numpy as np

from code.business_entity_resolution.src import model as model_mod
from code.business_entity_resolution.src.config import CANDIDATES_PER_S1
from code.business_entity_resolution.src.blocking import MultiChannelBlocker
from code.business_entity_resolution.src.features import (
    extract_features_from_candidates, FeatureExtractor)
from code.business_entity_resolution.src.model import TriEnsembleModel
from code.business_entity_resolution.src.threshold_optimizer_fast import VectorizedThresholdOptimizer

DATA = ROOT / "dataset"
CKPT = ROOT / "checkpoints"
OUT = ROOT / "output"
MOD = ROOT / "models"
for d in (CKPT, OUT, MOD):
    d.mkdir(exist_ok=True, parents=True)

NWORK = int(os.environ.get("NWORK", "8"))
SAMPLE_N = int(os.environ.get("SAMPLE_N", "50000"))
N_FOLDS = 3
SEED = 42
CHUNK = int(os.environ.get("INFER_CHUNK", "25000"))

def load_tsv(p):
    return pl.read_csv(p, separator="\t").to_pandas()

def shard(items, n):
    items = list(items)
    k = (len(items) + n - 1) // n
    return [items[i * k:(i + 1) * k] for i in range(n) if items[i * k:(i + 1) * k]]

def block_queries(s1_rec, s1_idx, pool_idx, pool_rec):
    """Threaded per-entity candidate generation (indices read-only)."""
    blk_holder = block_queries.blk
    def one(sh):
        out = {}
        for sid, rec in sh:
            scores = blk_holder.generate_candidates_for_entity(sid, rec, s1_idx, pool_idx, pool_rec)
            out[sid] = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:CANDIDATES_PER_S1]
        return out
    res = {}
    with ThreadPoolExecutor(max_workers=NWORK) as ex:
        for part in ex.map(one, shard(s1_rec.items(), NWORK)):
            res.update(part)
    return res

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

def featurize_shards(cands, s1_df, s2_df, s3_df, workers=None):
    """Threaded feature extraction over candidate shards."""
    workers = workers or int(os.environ.get("FEAT_WORK", "4"))
    import code.business_entity_resolution.src.features as F

    def one(sh):
        ex = __import__("code.business_entity_resolution.src.features",
                        fromlist=["FeatureExtractor"]).FeatureExtractor(verbose=False)
        sub = {k: cands[k] for k in sh}
        df = F.extract_features_from_candidates(sub, s1_df, s2_df, s3_df, phase=2, verbose=False)
        del sub
        gc.collect()
        return df
    keys = list(cands.keys())
    shards = shard(keys, max(workers * 2, 8))
    log(f"featurize: {len(keys):,} S1 in {len(shards)} shards x{workers} workers (RSS {rss_gb():.1f}GB)")
    parts = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, df in enumerate(ex.map(one, shards)):
            parts.append(df)
            log(f"featurize shard {i+1}/{len(shards)} done (RSS {rss_gb():.1f}GB)")
    out = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return out

# ============================ TRAIN ============================
th_path = CKPT / "thresholds.json"
if not th_path.exists():
    log("loading train files")
    s1 = load_tsv(DATA / "train" / "train_source1.tsv")
    s2 = load_tsv(DATA / "train" / "train_source2.tsv")
    s3 = load_tsv(DATA / "train" / "train_source3.tsv")
    gt = load_tsv(DATA / "train" / "train_ground_truth.tsv")
    log(f"s1={len(s1):,} s2={len(s2):,} s3={len(s3):,}")
    gt_dict = {}
    for sid, m in zip(gt["source1_entity_id"].values, gt["matched_entity_ids"].fillna("").values):
        gt_dict[sid] = set(m.split(",")) if m else set()
    del gt
    gc.collect()

    s1["_singleton"] = s1["entity_id"].map(lambda i: len(gt_dict.get(i, set())) == 0)
    parts = []
    for _, grp in s1.groupby(["country", "_singleton"]):
        n = max(1, round(len(grp) * SAMPLE_N / len(s1)))
        parts.append(grp.sample(n=min(n, len(grp)), random_state=SEED))
    samp = pd.concat(parts).sample(frac=1.0, random_state=SEED).head(SAMPLE_N)
    sample_ids = samp["entity_id"].tolist()
    log(f"sampled {len(sample_ids):,} S1")
    sample_gt = {i: gt_dict.get(i, set()) for i in sample_ids}

    # sampled pool: all true matches + 3 distractors per match
    rng = np.random.RandomState(SEED)
    pool_all = pd.concat([s2, s3]).reset_index(drop=True)
    pool_ids = set(pool_all["entity_id"].values)
    want = set()
    for v in sample_gt.values():
        want |= v
    n_dis = 3 * max(1, len(want))
    all_ids = pool_all["entity_id"].values
    extra = all_ids[rng.choice(len(all_ids), min(n_dis, len(all_ids)))]
    want |= set(extra.tolist())
    pool = pool_all[pool_all["entity_id"].isin(want)].reset_index(drop=True)
    log(f"sampled pool: {len(pool):,} records")
    del pool_all
    gc.collect()

    feat_file = CKPT / f"train_feat_{SAMPLE_N}.parquet"
    if not feat_file.exists():
        s1s = samp.drop(columns=["_singleton"]).reset_index(drop=True)
        s1_rec = {r["entity_id"]: r for r in s1s.to_dict("records")}
        pool_rec = {r["entity_id"]: r for r in pool.to_dict("records")}
        blk = MultiChannelBlocker(verbose=True)
        block_queries.blk = blk
        s1_idx, pool_idx = blk.build_all_channels(s1_rec, pool_rec)
        cands = block_queries(s1_rec, s1_idx, pool_idx, pool_rec)
        rec = blk.measure_recall(cands, sample_gt)
        log(f"train blocking recall={rec*100:.2f}%")
        del s1_rec, pool_rec, s1_idx, pool_idx
        gc.collect()
        pool_s2 = pool[pool["entity_id"].str.startswith("S2-")].reset_index(drop=True)
        pool_s3 = pool[pool["entity_id"].str.startswith("S3-")].reset_index(drop=True)
        del pool  # ponytail: free ~1M-row frame before feature peak
        gc.collect()
        feats = featurize_shards(cands, s1s, pool_s2, pool_s3)
        del cands, pool_s2, pool_s3
        gc.collect()
        lab = np.array([1 if c in sample_gt.get(s, set()) else 0
                        for s, c in zip(feats["s1_id"].values, feats["s2_s3_id"].values)],
                       dtype=np.int8)
        feats["label"] = lab
        for c in feats.columns:
            if feats[c].dtype == np.float64:
                feats[c] = feats[c].astype(np.float32)
        log(f"pairs={len(feats):,} pos={lab.sum():,}")
        feats.to_parquet(feat_file, index=False)
        del feats
        gc.collect()
    feats = pd.read_parquet(feat_file)
    log(f"train features: {len(feats):,} pairs")
    del s1, s2, s3, samp
    for _v in ("pool",):
        try:
            del globals()[_v]
        except KeyError:
            pass
    gc.collect()

    model_mod.N_SPLITS_CV = N_FOLDS
    groups = feats["s1_id"].map({v: i for i, v in enumerate(feats["s1_id"].unique())}).values
    X = feats.drop(columns=["label"])
    y = feats["label"].values
    log(f"training {N_FOLDS}-fold tri-ensemble on {len(X):,} pairs")
    ens = TriEnsembleModel(verbose=True)
    trained, oof = ens.train_groupkfold(X, y, groups)
    ens.save_models(trained, MOD)

    oof_sorted = oof.set_index("row_index").reindex(feats.index)
    probs = oof_sorted["blend_prob"].to_numpy(dtype=np.float64)
    s1_arr = feats["s1_id"].to_numpy()
    cand_arr = feats["s2_s3_id"].to_numpy()
    order = np.argsort(s1_arr, kind="stable")
    ss, cs, ps = s1_arr[order], cand_arr[order], probs[order]
    uniq, starts = np.unique(ss, return_index=True)
    counts = np.diff(np.append(starts, len(ss)))
    prob_by_s1 = {}
    for sid, st, cn in zip(uniq, starts, counts):
        sl = slice(st, st + cn)
        prob_by_s1[sid] = list(zip(cs[sl].tolist(), ps[sl].tolist()))
    opt = VectorizedThresholdOptimizer(beta=0.5)
    s_tau, m_tau, f, _ = opt.grid_search(prob_by_s1, sample_gt)
    log(f"BEST score={s_tau} margin={m_tau} macroF={f*100:.2f}%")
    th_path.write_text(json.dumps({"score_tau": s_tau, "margin_tau": m_tau,
                                   "macro_f": f, "sample_n": SAMPLE_N}))
    del feats, X, probs
    gc.collect()
else:
    log("thresholds exist, skipping train")

# ============================ INFERENCE ============================
th = json.loads(th_path.read_text())
SCORE_TAU, MARGIN_TAU = th["score_tau"], th["margin_tau"]
log(f"thresholds: score={SCORE_TAU} margin={MARGIN_TAU}")
opt = VectorizedThresholdOptimizer(beta=0.5)
ens = TriEnsembleModel(verbose=False)
models = ens.load_models(MOD)

s1_test = load_tsv(DATA / "test" / "test_source1.tsv")
s2_test = load_tsv(DATA / "test" / "test_source2.tsv")
s3_test = load_tsv(DATA / "test" / "test_source3.tsv")
log(f"test s1={len(s1_test):,} s2={len(s2_test):,} s3={len(s3_test):,}")

done_path = CKPT / "infer_done.json"
done = json.loads(done_path.read_text()) if done_path.exists() else {}
match_path = OUT / "matching_results.tsv"
cand_path = OUT / "candidate_pairs.tsv"
if not match_path.exists():
    match_path.write_text("source1_entity_id\tmatched_entity_ids\n")
    cand_path.write_text("source1_entity_id\tcandidate_entity_ids\n")

for country in ["India", "US", "France"]:
    s1c = s1_test[s1_test["country"] == country].reset_index(drop=True)
    if len(s1c) == 0:
        continue
    pool = pd.concat([s2_test[s2_test["country"] == country],
                      s3_test[s3_test["country"] == country]]).reset_index(drop=True)
    pool_s2 = pool[pool["entity_id"].str.startswith("S2-")].reset_index(drop=True)
    pool_s3 = pool[pool["entity_id"].str.startswith("S3-")].reset_index(drop=True)
    pool_rec = {r["entity_id"]: r for r in pool.to_dict("records")}
    del pool
    gc.collect()
    log(f"{country}: {len(s1c):,} S1")
    blk = MultiChannelBlocker(verbose=True)
    block_queries.blk = blk
    pn, pa = blk.compute_token_frequencies(pool_rec)
    pool_idx = {
        "channel_0": blk.build_channel_0_exact_core_name(pool_rec),
        "channel_1": blk.build_channel_1_name_tokens(pool_rec),
        "channel_2": blk.build_channel_2_address_tokens(pool_rec),
        "channel_3": blk.build_channel_3_street_number_street(pool_rec),
        "channel_4": blk.build_channel_4_postal_distinctive(pool_rec),
        "name_freq": pn, "addr_freq": pa}
    pool_n = len(pool_rec)
    # ponytail: per-entity blocking uses only len(pool_rec); 4.7M record dicts are pure overhead
    pool_rec = type("PoolN", (), {"__len__": lambda self: pool_n})()
    del pn, pa
    gc.collect()
    n_chunks = (len(s1c) + CHUNK - 1) // CHUNK
    for ci in range(n_chunks):
        key = f"{country}/{ci}"
        if key in done:
            continue
        chunk = s1c.iloc[ci * CHUNK:(ci + 1) * CHUNK].reset_index(drop=True)
        s1_rec = {r["entity_id"]: r for r in chunk.to_dict("records")}
        s1_idx = {
            "channel_0": blk.build_channel_0_exact_core_name(s1_rec),
            "channel_1": blk.build_channel_1_name_tokens(s1_rec),
            "channel_2": blk.build_channel_2_address_tokens(s1_rec),
            "channel_3": blk.build_channel_3_street_number_street(s1_rec),
            "channel_4": blk.build_channel_4_postal_distinctive(s1_rec),
            "name_freq": {}, "addr_freq": {}}
        cands = block_queries(s1_rec, s1_idx, pool_idx, pool_rec)
        feats = featurize_shards(cands, chunk, pool_s2, pool_s3)
        probs = ens.predict_ensemble(feats, models)
        prob_by_s1 = {}
        for sid, cid, p in zip(feats["s1_id"].values, feats["s2_s3_id"].values, probs):
            prob_by_s1.setdefault(sid, []).append((cid, float(p)))
        preds = opt.apply_thresholds(prob_by_s1, SCORE_TAU, MARGIN_TAU)
        with open(match_path, "a") as fm, open(cand_path, "a") as fc:
            for sid in chunk["entity_id"].values:
                fm.write(f"{sid}\t{','.join(sorted(preds.get(sid, set())))}\n")
                fc.write(f"{sid}\t{','.join(sorted(cid for cid, _ in cands.get(sid, [])))}\n")
        done[key] = True
        done_path.write_text(json.dumps(done))
        log(f"{country} chunk {ci+1}/{n_chunks} done")
        del s1_rec, s1_idx, cands, feats, probs, prob_by_s1, preds, chunk
        gc.collect()
    del pool_s2, pool_s3, pool_rec, pool_idx, s1c, pool_n
    # ponytail: shrink the full test frames as countries complete; peak was full-frames + indices
    s1_test = s1_test[s1_test["country"] != country].reset_index(drop=True)
    s2_test = s2_test[s2_test["country"] != country].reset_index(drop=True)
    s3_test = s3_test[s3_test["country"] != country].reset_index(drop=True)
    gc.collect()
log("INFERENCE DONE")
