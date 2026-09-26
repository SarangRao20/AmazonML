"""Sparse TF-IDF multi-channel blocking.

Why this replaces the dict-based blocker
---------------------------------------
The previous blocker accumulated candidate scores with Python dict lookups
and set unions, which is O(entities) interpreted work. Retrieving for
259,452 Source 1 entities against a 1.43M pool took over 15 minutes and had
not finished, and the per-entity candidate cap had to be cut to 12 just to
get a result out. That is the wrong trade: the cap is the recall ceiling.

Reference project `ayan_multiview` measures union pair recall of 0.9905 at
112 candidates per S1 and 0.989 at 45, against 0.9518 at 35 for the dict
blocker. The fix is to stop interpreting the retrieval loop and let a
compiled sparse matrix product do it, which buys the candidate cap back.

`sparse_dot_topn.sp_matmul_topn` is the same primitive used by reference
projects `vitthalg17` (0.9545 verified on the public leaderboard) and
`ayan_multiview`.

Field-prefixed documents
------------------------
Name, address and number tokens are prefixed into one bag
(``n:tenitech n:limited a:liberty a:church d:588``) so a single TF-IDF
space scores each field separately while still allowing a combined channel.
Without prefixes, a name token and an address token of the same string
would be indistinguishable and the address channel would leak into the name
channel.

IDF is fitted per country, on that country's own records and without
labels, so an unseen country such as France gets its own statistics rather
than inheriting US/India ones.
"""

import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from .normalize import normalize_address, normalize_name, extract_numeric_signature

# Retrieval-side knobs. max_df drops near-useless very common terms, which
# both speeds up top-K and stops generic tokens flooding the candidate set.
CHUNK = int(os.environ.get("ER_BLOCK_CHUNK", 200_000))
MAX_DF_FRAC = float(os.environ.get("ER_MAX_DF_FRAC", 0.01))
MAX_DF_MIN = 2000
N_THREADS = int(os.environ.get("ER_THREADS", os.cpu_count() or 4))

# weight per channel when unioning candidates; agreement across channels is
# a strong signal so the exact-name channel is weighted highest.
CHANNEL_WEIGHTS = {
    "name_word": 1.0,
    "name_char": 1.0,
    "addr": 1.0,
    "combo": 1.0,
    "exact": 3.0,
}
DEFAULT_K = {
    "name_word": 10,
    "name_char": 10,
    "addr": 10,
    "combo": 10,
    "exact": 10,
}


def _doc(record: dict) -> str:
    """Field-prefixed bag of tokens for one record."""
    name = normalize_name(record.get("business_name", "") or "")
    addr = normalize_address(record.get("business_address", "") or "")
    parts = [f"n:{t}" for t in name.split()]
    parts += [f"a:{t}" for t in addr.split()]
    parts += [f"d:{n}" for n in extract_numeric_signature(
        record.get("business_address", "") or "")]
    return " ".join(parts) if parts else "__empty__"


def _name_only(record: dict) -> str:
    name = normalize_name(record.get("business_name", "") or "")
    return name if name else "__empty__"


def _addr_only(record: dict) -> str:
    addr = normalize_address(record.get("business_address", "") or "")
    return addr if addr else "__empty__"


def _combo(record: dict) -> str:
    name = normalize_name(record.get("business_name", "") or "")
    addr = normalize_address(record.get("business_address", "") or "")
    return f"{name} {addr}".strip() or "__empty__"


def _exact_key(record: dict) -> str:
    """Order-invariant core-name key: 'galaxy properties' == 'properties galaxy'."""
    name = normalize_name(record.get("business_name", "") or "")
    toks = sorted(t for t in name.split() if len(t) > 2)
    return " ".join(toks) if toks else name


VECTORISERS = {
    "name_word": lambda max_df: TfidfVectorizer(
        token_pattern=r"n:\S+", ngram_range=(1, 2), min_df=1, max_df=max_df,
        sublinear_tf=True, lowercase=False, dtype=np.float32),
    "name_char": lambda max_df: TfidfVectorizer(
        input="content", analyzer="char_wb", ngram_range=(3, 4), min_df=2,
        max_df=max_df, sublinear_tf=True, lowercase=False, dtype=np.float32,
        preprocessor=lambda s: s.replace("n:", "").replace("a:", "")
        .replace("d:", "").strip()),
    "addr": lambda max_df: TfidfVectorizer(
        token_pattern=r"a:\S+", ngram_range=(1, 2), min_df=2, max_df=max_df,
        sublinear_tf=True, lowercase=False, dtype=np.float32),
    "combo": lambda max_df: TfidfVectorizer(
        token_pattern=r"\S+", ngram_range=(1, 1), min_df=1, max_df=max_df,
        sublinear_tf=True, lowercase=False, dtype=np.float32),
}


class SparseBlocker:
    """Multi-channel TF-IDF retrieval within a single country.

    Country partitioning is lossless rather than a heuristic: across all
    7,638,365 matched id occurrences in train_ground_truth.tsv, none crosses
    a country boundary, so no true match can be lost by indexing and
    retrieving each country separately. It is also what keeps peak memory
    at the largest single country instead of the whole pool.
    """

    def __init__(self, k: Optional[Dict[str, int]] = None,
                 verbose: bool = False) -> None:
        self.k = dict(DEFAULT_K)
        if k:
            self.k.update(k)
        self.verbose = verbose

    def _channel_matrix(self, docs: List[str], channel: str, max_df: float):
        if channel == "name_char":
            text = [d.replace("n:", " ").replace("a:", " ")
                     .replace("d:", " ").strip() or "__empty__" for d in docs]
        else:
            text = docs
        vec = VECTORISERS[channel](max_df)
        mat = vec.fit_transform(text)
        return vec, mat.tocsr()

    def candidates_for_country(
        self,
        s1_records: Dict[str, dict],
        pool_records: Dict[str, dict],
        max_candidates: int = 45,
        min_score: float = 0.0,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Retrieve up to ``max_candidates`` targets per S1 entity."""
        s1_ids = list(s1_records)
        pool_ids = list(pool_records)
        if not s1_ids or not pool_ids:
            return {}

        s1_docs = [_doc(s1_records[i]) for i in s1_ids]
        pool_docs = [_doc(pool_records[i]) for i in pool_ids]
        n_s1, n_pool = len(s1_ids), len(pool_ids)
        max_df = max(MAX_DF_MIN, int(MAX_DF_FRAC * n_s1))

        # channel -> {s1_row: [(pool_row, score)]}
        retrieved: Dict[str, Dict[int, List[Tuple[int, float]]]] = defaultdict(dict)
        for channel in ("name_word", "name_char", "addr", "combo"):
            k = self.k[channel]
            if k <= 0:
                continue
            t0 = time.time()
            if channel == "name_char":
                s1_txt = [d.replace("n:", " ").replace("a:", " ")
                          .replace("d:", " ").strip() or "__empty__" for d in s1_docs]
                pool_txt = [d.replace("n:", " ").replace("a:", " ")
                            .replace("d:", " ").strip() or "__empty__" for d in pool_docs]
            else:
                s1_txt, pool_txt = s1_docs, pool_docs
            vec, B = self._channel_matrix(pool_txt, channel, max_df)
            Q = vec.transform(s1_txt).tocsr()
            Q = Q[:, :].astype(np.float32)
            for start in range(0, n_s1, CHUNK):
                stop = min(start + CHUNK, n_s1)
                top = sp_matmul_topn(
                    Q[start:stop], B, top_n=min(k, n_pool),
                    sort=True, n_threads=N_THREADS)
                # top is a CSR whose rows are the chunk's S1 records; each
                # stored (col=B_row, value=cosine) is one retrieved target.
                indptr, indices, data = top.indptr, top.indices, top.data
                for r in range(stop - start):
                    lo, hi = indptr[r], indptr[r + 1]
                    if lo == hi:
                        continue
                    retrieved[channel].setdefault(start + r, []).extend(
                        (int(c), float(v))
                        for c, v in zip(indices[lo:hi], data[lo:hi]))
            if self.verbose:
                print(f"      {channel}: {time.time() - t0:.0f}s "
                      f"vocab={B.shape[1]:,}", flush=True)

        # exact order-invariant core-name key, capped per group
        t0 = time.time()
        exact_groups: Dict[str, List[int]] = defaultdict(list)
        for j, pid in enumerate(pool_ids):
            exact_groups[_exact_key(pool_records[pid])].append(j)
        for i, sid in enumerate(s1_ids):
            key = _exact_key(s1_records[sid])
            members = exact_groups.get(key)
            if members and len(members) <= 200:
                retrieved["exact"].setdefault(i, []).extend(
                    (j, 1.0) for j in members)
        if self.verbose:
            print(f"      exact: {time.time() - t0:.0f}s", flush=True)

        # union with per-channel weights, then cap
        out: Dict[str, List[Tuple[str, float]]] = {}
        for i, sid in enumerate(s1_ids):
            scores: Dict[int, float] = defaultdict(float)
            for channel, per_row in retrieved.items():
                if i not in per_row:
                    continue
                w = CHANNEL_WEIGHTS[channel]
                for j, v in per_row[i]:
                    scores[j] += w * v
            if not scores:
                out[sid] = []
                continue
            ranked = sorted(scores.items(), key=lambda t: (-t[1], pool_ids[t[0]]))
            ranked = [(j, v) for j, v in ranked if v > min_score][:max_candidates]
            out[sid] = [(pool_ids[j], v) for j, v in ranked]
        return out


def blocking_recall(candidates: Dict[str, List[Tuple[str, float]]],
                    ground_truth: Dict[str, Set[str]]) -> Dict[str, float]:
    """Pair recall and entity full coverage over the retrieved candidate set."""
    total = captured = 0
    non_single = full = 0
    for s1_id, truth in ground_truth.items():
        if not truth:
            continue
        non_single += 1
        got = {c for c, _ in candidates.get(s1_id, [])}
        total += len(truth)
        captured += len(truth & got)
        if truth.issubset(got):
            full += 1
    return {
        "pair_recall": captured / total if total else 0.0,
        "entity_full_coverage": full / non_single if non_single else 0.0,
        "true_pairs": total,
        "non_singletons": non_single,
    }
