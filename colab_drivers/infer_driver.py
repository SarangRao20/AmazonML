"""Chunked test inference. Per country: build pool index once, stream S1 in
chunks, block -> featurize -> ensemble predict -> thresholds -> append TSVs.
Outputs + resume state live on Drive (preemption-safe).
Run:  nohup python3 infer_driver.py > ckpt/infer.log 2>&1 &
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

from code.business_entity_resolution.src.config import CANDIDATES_PER_S1
from code.business_entity_resolution.src.blocking import MultiChannelBlocker
from code.business_entity_resolution.src.features import extract_features_from_candidates
from code.business_entity_resolution.src.model import TriEnsembleModel
from code.business_entity_resolution.src.threshold_optimizer_fast import VectorizedThresholdOptimizer

DRIVE = Path("/content/drive/MyDrive/AmazonML")
DATA = DRIVE / "dataset"
CKPT = DRIVE / "checkpoints"
OUT = DRIVE / "output"
OUT.mkdir(exist_ok=True, parents=True)
CHUNK = int(os.environ.get("INFER_CHUNK", "100000"))

th = json.loads((CKPT / "thresholds.json").read_text())
SCORE_TAU, MARGIN_TAU = th["score_tau"], th["margin_tau"]
log(f"thresholds: score={SCORE_TAU} margin={MARGIN_TAU}")
opt = VectorizedThresholdOptimizer(beta=0.5)
ens = TriEnsembleModel(verbose=False)
models = ens.load_models(CKPT / "models")
log("models loaded")

def load_tsv(path):
    return pl.read_csv(path, separator="\t").to_pandas()

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
    log(f"{country}: {len(s1c):,} S1 vs pool {len(pool):,}")

    # pool-side indices built ONCE per country
    blk = MultiChannelBlocker(verbose=True)
    pname_freq, paddr_freq = blk.compute_token_frequencies(pool_rec)
    pool_idx = {
        "channel_0": blk.build_channel_0_exact_core_name(pool_rec),
        "channel_1": blk.build_channel_1_name_tokens(pool_rec),
        "channel_2": blk.build_channel_2_address_tokens(pool_rec),
        "channel_3": blk.build_channel_3_street_number_street(pool_rec),
        "channel_4": blk.build_channel_4_postal_distinctive(pool_rec),
        "name_freq": pname_freq, "addr_freq": paddr_freq,
    }
    n_chunks = (len(s1c) + CHUNK - 1) // CHUNK
    for ci in range(n_chunks):
        key = f"{country}/{ci}"
        if key in done:
            continue
        chunk = s1c.iloc[ci * CHUNK:(ci + 1) * CHUNK]
        s1_rec = {r["entity_id"]: r for r in chunk.to_dict("records")}
        # S1-side indices for this chunk only
        s1_idx = {
            "channel_0": blk.build_channel_0_exact_core_name(s1_rec),
            "channel_1": blk.build_channel_1_name_tokens(s1_rec),
            "channel_2": blk.build_channel_2_address_tokens(s1_rec),
            "channel_3": blk.build_channel_3_street_number_street(s1_rec),
            "channel_4": blk.build_channel_4_postal_distinctive(s1_rec),
            "name_freq": {}, "addr_freq": {},
        }
        cands = {}
        for sid, rec in s1_rec.items():
            scores = blk.generate_candidates_for_entity(sid, rec, s1_idx, pool_idx, pool_rec)
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:CANDIDATES_PER_S1]
            cands[sid] = top
        feats = extract_features_from_candidates(cands, chunk, pool_s2, pool_s3,
                                                 phase=2, verbose=False)
        probs = ens.predict_ensemble(feats, models)
        prob_by_s1 = {}
        for sid, cid, p in zip(feats["s1_id"].values, feats["s2_s3_id"].values, probs):
            prob_by_s1.setdefault(sid, []).append((cid, float(p)))
        preds = opt.apply_thresholds(prob_by_s1, SCORE_TAU, MARGIN_TAU)
        with open(match_path, "a") as fm, open(cand_path, "a") as fc:
            for sid in chunk["entity_id"].values:
                m = sorted(preds.get(sid, set()))
                c = sorted(cid for cid, _ in cands.get(sid, []))
                fm.write(f"{sid}\t{','.join(m)}\n")
                fc.write(f"{sid}\t{','.join(c)}\n")
        done[key] = True
        done_path.write_text(json.dumps(done))
        log(f"{country} chunk {ci+1}/{n_chunks} done ({len(chunk):,} S1)")
        del s1_rec, s1_idx, cands, feats, probs, prob_by_s1, preds
        gc.collect()
    del pool, pool_s2, pool_s3, pool_rec, pool_idx, s1c
    gc.collect()
log("INFERENCE DONE")
