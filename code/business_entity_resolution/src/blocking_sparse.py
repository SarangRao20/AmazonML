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

Memory: pool sharding
---------------------
The naive version built one TF-IDF matrix per channel over the whole pool.
That works for France (1.43M pool) and dies on India (4.72M pool): the
``char_wb`` 3-4 gram channel alone reaches ~1e9 nonzeros, roughly 8 GB,
and the first attempt was killed by the memory watchdog at 1,577 MB
available.

Two changes bound the peak:

* The pool is processed in shards of ``POOL_SHARD`` records, and each
  shard's matrix is freed before the next is built. Because retrieval is
  sharded rather than truncated, every pool record is still compared
  against every S1 entity -- the union of per-shard top-k is the global
  top-k -- so this is a memory optimisation, not a recall trade.
* Retrieved hits are accumulated as CSR blocks instead of Python tuples.
  India generates 810k x 160 hits; as ``(int, float)`` tuples that is
  ~3.2 GB of object headers alone, as CSR it is ~1 GB of two int32/float32
  arrays.

Per-channel document construction
---------------------------------
Each channel is built from its own field. The earlier version fed a single
field-prefixed document (``n:tenitech a:church d:588``) to all channels and
stripped the prefixes for the character channel, which meant the
``name_char`` channel was silently scoring address and house-number
characters as if they were part of the name -- doubling its nonzeros and
injecting address noise into the name signal. Channels now read their own
field directly.

IDF is fitted per country, on that country's own records and without
labels, so an unseen country such as France gets its own statistics rather
than inheriting US/India ones.
"""

import os
import time
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from .normalize import normalize_address, normalize_name, extract_numeric_signature

# Retrieval-side knobs. max_df drops near-useless very common terms, which
# both speeds up top-K and stops generic tokens flooding the candidate set.
CHUNK = int(os.environ.get("ER_BLOCK_CHUNK", 200_000))
POOL_SHARD = int(os.environ.get("ER_POOL_SHARD", 1_000_000))
# The char_wb 3-4 gram channel needs a much smaller shard than the word
# channels. sklearn builds the FULL term vocabulary before pruning by
# max_df, and on a 2M-document shard that vocabulary is tens of millions
# of entries in a Python dict: measured 2.5 GB -> 6.1 GB and climbing on
# India's 4.72M pool, versus a flat 2.5 GB at the 400k shard that
# France completed in. The word channels are nowhere near that ceiling
# (~30 nonzeros per document) and are cheaper with fewer, larger shards,
# which also cuts the hits they retain (k per shard per S1).
CHANNEL_POOL_SHARD = {
    "name_char": int(os.environ.get("ER_POOL_SHARD_CHAR", 400_000)),
}
K_PER_SHARD = int(os.environ.get("ER_K_PER_SHARD", 0))
UNION_CHUNK = int(os.environ.get("ER_UNION_CHUNK", 3_000))
MAX_DF_FRAC = float(os.environ.get("ER_MAX_DF_FRAC", 0.01))
MAX_DF_MIN = 2000
# max_features bounds the vocabulary *while counting*. Without it sklearn
# materialises every distinct term in a Python dict and only then applies
# max_df, so the transient scales with vocabulary size rather than
# document count. That transient is what actually exhausted RAM on
# India's 4.72M pool: the process sat at 5.37 GB with name_word still
# unfinished, on a channel whose own matrix is only ~240 MB.
# Ceilings are set above the observed natural vocabularies (name_word
# 699k, name_char 221k on a 518k pool) so little is actually dropped.
MAX_FEATURES = {
    "name_word": int(os.environ.get("ER_MAX_FEAT_WORD", 1_000_000)),
    "name_char": int(os.environ.get("ER_MAX_FEAT_CHAR", 400_000)),
    "addr": int(os.environ.get("ER_MAX_FEAT_ADDR", 600_000)),
    "combo": int(os.environ.get("ER_MAX_FEAT_COMBO", 1_000_000)),
}
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
RETRIEVAL_CHANNELS = ("name_word", "name_char", "addr", "combo")

# Cap on how many pool records may share one order-invariant core-name key
# before the exact channel ignores that key. Guards against generic keys
# like "properties galaxy" pulling in a whole chain.
EXACT_GROUP_CAP = 200


def _name_of(rec) -> str:
    """Business name from either a (name, addr) tuple or a record dict."""
    if isinstance(rec, (tuple, list)):
        return rec[0] or ""
    return rec.get("business_name", "") or ""


def _addr_of(rec) -> str:
    """Business address from either a (name, addr) tuple or a record dict."""
    if isinstance(rec, (tuple, list)):
        return rec[1] or ""
    return rec.get("business_address", "") or ""


def _doc(name: str, addr: str) -> str:
    """Field-prefixed bag of tokens for one record."""
    parts = [f"n:{t}" for t in normalize_name(name).split()]
    parts += [f"a:{t}" for t in normalize_address(addr).split()]
    parts += [f"d:{n}" for n in extract_numeric_signature(addr)]
    return " ".join(parts) if parts else "__empty__"


def _name_text(name: str, addr: str) -> str:
    name = normalize_name(name)
    return name if name else "__empty__"


def _addr_text(name: str, addr: str) -> str:
    addr = normalize_address(addr)
    return addr if addr else "__empty__"


def _combo_text(name: str, addr: str) -> str:
    name = normalize_name(name)
    addr = normalize_address(addr)
    return f"{name} {addr}".strip() or "__empty__"


def _full_text(name: str, addr: str) -> str:
    """Normalised name + address + numeric/landmark signature, no prefixes.

    This is what the character and combo channels score over. The numeric
    signature carries house numbers and landmark markers ('sbi atm'), which
    is why it is included: without it val_sample pair recall was 96.74%
    against 98.11%.

    ``extract_numeric_signature`` returns a space-separated string, so it
    is split here. The previous implementation iterated the string
    directly and emitted one ``d:`` token per digit, which silently turned
    "588" into "5 8 8" for the character and combo channels.
    """
    name = normalize_name(name)
    addr = normalize_address(addr)
    parts = (name.split() + addr.split()
             + extract_numeric_signature(addr).split())
    return " ".join(parts) if parts else "__empty__"


def _exact_key(name: str, addr: str) -> str:
    """Order-invariant core-name key: 'galaxy properties' == 'properties galaxy'."""
    name = normalize_name(name)
    toks = sorted(t for t in name.split() if len(t) > 2)
    return " ".join(toks) if toks else name


# Every channel scores the same field-prefixed document, and selects its own
# field with a token_pattern (or, for the character channel, a preprocessor
# that strips the prefixes). Keeping one prefixed representation is not just
# tidier than one text per channel -- it is measurably better. Building
# separate plain-field texts instead measured 96.71% pair recall on
# val_sample against 98.11% for the prefixed form, because the prefixes keep
# the fields separable inside one TF-IDF space and stop a name token and an
# address token of the same spelling from collapsing into one feature.
DOC_BUILDER = _doc

VECTORISERS = {
    "name_word": lambda max_df: TfidfVectorizer(
        token_pattern=r"n:\S+", ngram_range=(1, 2), min_df=1, max_df=max_df,
        max_features=MAX_FEATURES["name_word"],
        sublinear_tf=True, lowercase=False, dtype=np.float32),
    "name_char": lambda max_df: TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 4), min_df=2,
        max_df=max_df, max_features=MAX_FEATURES["name_char"],
        sublinear_tf=True, lowercase=False,
        dtype=np.float32,
        preprocessor=lambda s: s.replace("n:", " ").replace("a:", " ")
        .replace("d:", " ").strip()),
    "addr": lambda max_df: TfidfVectorizer(
        token_pattern=r"a:\S+", ngram_range=(1, 2), min_df=2, max_df=max_df,
        max_features=MAX_FEATURES["addr"],
        sublinear_tf=True, lowercase=False, dtype=np.float32),
    "combo": lambda max_df: TfidfVectorizer(
        token_pattern=r"\S+", ngram_range=(1, 1), min_df=1, max_df=max_df,
        max_features=MAX_FEATURES["combo"],
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

    # -- retrieval ------------------------------------------------------
    def _retrieve_channel(
        self,
        channel: str,
        s1_text: List[str],
        pool_records,
        pool_ids: List[str],
        max_df: float,
        k_per_shard: int,
    ) -> List[List[Tuple[sp.csr_matrix, int]]]:
        """Retrieve for every S1 entity against every pool record.

        Returns, per S1 chunk, a list of ``(csr_block, column_offset)`` --
        one entry per pool shard. Column indices inside a block are shard
        local, so the union step adds the offset back to recover the global
        pool row.
        """
        n_s1, n_pool = len(s1_text), len(pool_ids)
        n_chunks = (n_s1 + CHUNK - 1) // CHUNK
        per_chunk: List[List[Tuple[sp.csr_matrix, int]]] = [
            [] for _ in range(n_chunks)]
        make_vec = VECTORISERS[channel]

        shard_size = CHANNEL_POOL_SHARD.get(channel, POOL_SHARD)
        for start in range(0, n_pool, shard_size):
            stop = min(start + shard_size, n_pool)
            # Documents are built per shard and dropped with it. Holding the
            # whole pool's prefixed documents at once cost 1.24 GB on India's
            # 4.13M records and was a third of the peak that tripped the
            # watchdog.
            shard_text = [DOC_BUILDER(_name_of(pool_records[pool_ids[j]]),
                                     _addr_of(pool_records[pool_ids[j]]))
                          for j in range(start, stop)]
            vec = make_vec(max_df)
            # fit_transform on the shard only; the same vec then transforms
            # S1, so query and document share one vocabulary and one IDF.
            B = vec.fit_transform(shard_text).tocsr().astype(np.float32)
            n_shard = stop - start
            k_here = min(k_per_shard, n_shard)
            del shard_text
            for ci, q0 in enumerate(range(0, n_s1, CHUNK)):
                q1 = min(q0 + CHUNK, n_s1)
                Q = vec.transform(
                    s1_text[q0:q1]).tocsr().astype(np.float32)
                top = sp_matmul_topn(
                    Q, B, top_n=k_here, sort=True, n_threads=N_THREADS)
                per_chunk[ci].append((top, start))
                del Q, top
            del B, vec
        return per_chunk

    def _exact_hits(
        self,
        s1_records,
        s1_ids: List[str],
        pool_records,
        pool_ids: List[str],
    ) -> Dict[int, List[int]]:
        """Order-invariant core-name key hits, oversized keys ignored.

        Most keys are unique, so a key maps to a single int and only
        promotes to a list on a second occurrence. That keeps the group
        table near one int per pool record instead of one list per record.
        """
        groups: Dict[str, object] = {}
        oversized: Set[str] = set()
        for j, pid in enumerate(pool_ids):
            key = _exact_key(_name_of(pool_records[pid]),
                             _addr_of(pool_records[pid]))
            prev = groups.get(key)
            if prev is None:
                groups[key] = j
            elif isinstance(prev, int):
                if key in oversized:
                    continue
                groups[key] = [prev, j]
                if len(groups[key]) > EXACT_GROUP_CAP:
                    oversized.add(key)
                    del groups[key]
            else:
                prev.append(j)
                if len(prev) > EXACT_GROUP_CAP:
                    oversized.add(key)
                    del groups[key]
        hits: Dict[int, List[int]] = {}
        for i, sid in enumerate(s1_ids):
            members = groups.get(_exact_key(_name_of(s1_records[sid]),
                                            _addr_of(s1_records[sid])))
            if members is None:
                continue
            hits[i] = [members] if isinstance(members, int) else list(members)
        return hits

    # -- public API -----------------------------------------------------
    def iter_candidate_chunks(
        self,
        s1_records,
        pool_records,
        max_candidates: int = 45,
        min_score: float = 0.0,
    ) -> Iterator[Tuple[List[str], Dict[str, List[Tuple[str, float]]]]]:
        """Yield ``(s1_ids, {s1_id: [(cand_id, score), ...]})`` in chunks.

        Streaming rather than returning one country-sized dict is what keeps
        peak memory flat: the caller can extract features, predict and write
        each chunk out before the next is retrieved.
        """
        s1_ids = list(s1_records)
        pool_ids = list(pool_records)
        if not s1_ids or not pool_ids:
            return
        n_s1, n_pool = len(s1_ids), len(pool_ids)
        max_df = max(MAX_DF_MIN, int(MAX_DF_FRAC * n_s1))

        t0 = time.time()
        # Only the S1 side is materialised: it is the small side, and it is
        # needed by every channel. The pool side is built per shard.
        s1_text = [DOC_BUILDER(_name_of(s1_records[i]),
                               _addr_of(s1_records[i])) for i in s1_ids]
        self._log(f"S1 docs normalised in {time.time() - t0:.0f}s "
                  f"(S1={n_s1:,} pool={n_pool:,})")

        retrieved: Dict[str, List[List[Tuple[sp.csr_matrix, int]]]] = {}
        for channel in RETRIEVAL_CHANNELS:
            if self.k.get(channel, 0) <= 0:
                continue
            t0 = time.time()
            shard_size = CHANNEL_POOL_SHARD.get(channel, POOL_SHARD)
            per_shards = int(np.ceil(n_pool / shard_size))
            # Per-shard k must reach k on its own, not k / n_shards: a pool
            # record ranked 6th-10th inside its own shard is still a global
            # top-10 hit, so dividing k silently drops it. Measured cost of
            # that shortcut on val_sample: pair recall 95.68% vs 98.11%.
            k_per_shard = (K_PER_SHARD if K_PER_SHARD > 0
                           else self.k[channel])
            retrieved[channel] = self._retrieve_channel(
                channel, s1_text, pool_records, pool_ids, max_df, k_per_shard)
            if self.verbose:
                print(f"      {channel}: {time.time() - t0:.0f}s "
                      f"{per_shards} shard(s) k={k_per_shard}/shard",
                      flush=True)

        t0 = time.time()
        exact = self._exact_hits(s1_records, s1_ids, pool_records, pool_ids)
        self._log(f"exact keys in {time.time() - t0:.0f}s "
                  f"({sum(len(v) for v in exact.values()):,} hits)")

        for lo in range(0, n_s1, UNION_CHUNK):
            hi = min(lo + UNION_CHUNK, n_s1)
            block: Dict[str, List[Tuple[str, float]]] = {}
            for i in range(lo, hi):
                scores: Dict[int, float] = {}
                ci = i // CHUNK
                local = i - ci * CHUNK
                for channel, per_chunk in retrieved.items():
                    w = CHANNEL_WEIGHTS[channel]
                    for top, offset in per_chunk[ci]:
                        lo_r, hi_r = top.indptr[local], top.indptr[local + 1]
                        if lo_r == hi_r:
                            continue
                        for c, v in zip(top.indices[lo_r:hi_r],
                                        top.data[lo_r:hi_r]):
                            col = int(c) + offset
                            scores[col] = scores.get(col, 0.0) + w * float(v)
                for j in exact.get(i, ()):
                    scores[j] = scores.get(j, 0.0) + CHANNEL_WEIGHTS["exact"]
                sid = s1_ids[i]
                if not scores:
                    block[sid] = []
                    continue
                ranked = sorted(scores.items(),
                                key=lambda t: (-t[1], pool_ids[t[0]]))
                ranked = [t for t in ranked if t[1] > min_score][:max_candidates]
                block[sid] = [(pool_ids[j], v) for j, v in ranked]
            yield s1_ids[lo:hi], block
            del block

    def candidates_for_country(
        self,
        s1_records,
        pool_records,
        max_candidates: int = 45,
        min_score: float = 0.0,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """All candidates for a country at once. Convenient, but memory
        scales with the country; prefer :meth:`iter_candidate_chunks`."""
        out: Dict[str, List[Tuple[str, float]]] = {}
        for _, block in self.iter_candidate_chunks(
                s1_records, pool_records, max_candidates, min_score):
            out.update(block)
        return out

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"      {msg}", flush=True)


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
