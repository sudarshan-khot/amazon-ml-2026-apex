"""Blocking / candidate generation -- single entry point for Step 3.

Run from code/business_entity_resolution/ (after data_cleaning.py and make_folds.py):

    python src/blocking.py --data ../../dataset

Design (every choice was measured on the training pairs; see README / documentation):
  1. KEYS.  Single words are the wrong unit (most business names are combinations of common
     words).  What identifies a record is a RARE COMBINATION, so every record is turned into
     composite keys of 11 families (name word pairs, name word x address word, house number x
     street word, name word x state, ...).  A key's weight is its rarity in the Source-2/3 pool
     of the same country: w = log(N_pool / df).  Keys more frequent than a family cap are dropped.
  2. GENERATE.  For each Source-1 record, every pool record sharing >= 1 key is scored and the
     union of three short lists is kept:
       - top 40 by weighted Jaccard  (shared key weight / union key weight: sparse records that
         match on everything they have are not beaten by rich neighbours)
       - top 20 by raw shared weight
       - top 10 by name-only weight   (records whose address is empty or wrong)
  3. RE-RANK.  A small gradient-boosted model scores every generated pair from language-neutral
     features (scores, ranks, distance to the entity's best candidate).  It is CROSS-FITTED:
     a train entity is always re-ranked by a model that never saw its fold; test uses a model
     trained on all training folds.  (Trained on one country it transfers to the other, so it
     is safe for France, which has no training data.)
  4. KEEP the top K (default 20) per Source-1 record -> candidate pairs.

Outputs (artifacts/blocking/):
    train_candidates.tsv / test_candidates.tsv   one row per candidate pair (+ blocking features)
    candidate_pairs.tsv                          test candidates in the submission format
    rerankers.pkl                                the re-ranking models
    blocking_report.txt                          recall / F0.5 ceiling on train (out-of-fold)
Everything is computed per country: the ground truth has no cross-country matches.
"""
import argparse
import csv
import gc
import os
import pickle
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLEAN = os.path.join(ROOT, "artifacts", "cleaned_dataset")
OUT = os.path.join(ROOT, "artifacts", "blocking")

COLS = ["entity_id", "country", "name_core", "name_phon", "name_concat",
        "addr_words", "addr_nums", "house_no", "state_inf"]

# family -> (df cap, weight).  Keys more frequent than the cap never create a candidate; the cap
# is generous because the rarity weight already makes common keys cheap in the ranking.
FAMILIES = {
    "name_rare":    (100, 1.0),    # a single rare name word ('rzf', 'faison')
    "name_pair":    (1000, 1.0),   # two name words, order-free
    "phon_pair":    (1000, 1.0),   # two phonetic name words (typos, transliteration)
    "name_x_state": (1000, 1.0),   # name word inside a state
    "name_x_addr":  (1000, 1.0),   # name word + address word
    "concat":       (1000, 1.0),   # whole name glued (domains, handles)
    "hn_x_addr":    (1000, 1.0),   # house number + address word
    "num_x_addr":   (1000, 1.0),   # any address number + address word
    "num_x_state":  (1000, 1.0),   # address number inside a state
    "addr_pair":    (1000, 1.0),   # two address words (very common words excluded)
    "addr_x_state": (1000, 1.0),   # address word inside a state
}
NAME_FAMILIES = {"name_rare", "name_pair", "phon_pair", "name_x_state", "concat"}
ADDR_PAIR_MAX_SHARE = 0.002        # address words in > 0.2 % of the pool are not paired
LISTS = {"k_wj": 40, "k_score": 20, "k_name": 10}
RERANK_FEATS = ["score", "wj", "name_score", "nk", "r_score", "r_wj", "r_name", "score_rel", "wj_rel",
                "name_score_rel", "score_gap", "wj_gap", "name_score_gap", "n_cand", "name_share"]
OUT_COLS = ["source1_entity_id", "candidate_entity_id", "country", "fold", "blk_score", "blk_wj",
            "blk_name_score", "blk_nkeys", "rr_prob", "rr_rank"]


# ========================================================================================
# loading
# ========================================================================================
def load_clean(split, k, clean_dir=CLEAN, columns=COLS):
    path = os.path.join(clean_dir, split, f"{split}_source{k}.tsv")
    return pacsv.read_csv(
        path,
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in columns},
                                             include_columns=columns, strings_can_be_null=False,
                                             quoted_strings_can_be_null=False),
        read_options=pacsv.ReadOptions(block_size=1 << 26)).combine_chunks()


def to_np(a):
    return (a.combine_chunks() if isinstance(a, pa.ChunkedArray) else a).to_numpy(zero_copy_only=False)


def read_tsv(path, usecols=None):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, usecols=usecols)


# ========================================================================================
# token encoding (S1 and pool share one id space per column)
# ========================================================================================
def _flat(col, explode):
    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    if explode:
        lists = pc.split_pattern(col, " ")
        flat, rec = pc.list_flatten(lists), pc.list_parent_indices(lists)
    else:
        flat, rec = col, pa.array(np.arange(len(col), dtype=np.int64))
    keep = pc.not_equal(flat, "")
    return to_np(pc.filter(rec, keep)).astype(np.int64), pc.filter(flat, keep)


def encode_tokens(s1_col, pool_col, explode=True):
    """-> (s1_rec, s1_tok), (pool_rec, pool_tok), vocabulary size; (rec, tok) deduplicated and
    sorted by record then token"""
    r1, f1 = _flat(s1_col, explode)
    r2, f2 = _flat(pool_col, explode)
    enc = pc.dictionary_encode(pa.concat_arrays([f1, f2]))
    ids = enc.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    V = len(enc.dictionary)
    return _dedup(r1, ids[:len(f1)]), _dedup(r2, ids[len(f1):]), V


def encode_attr(s1_col, pool_col):
    """record-level attribute (state, house number) -> ids (-1 where empty)"""
    s1_col = s1_col.combine_chunks() if isinstance(s1_col, pa.ChunkedArray) else s1_col
    pool_col = pool_col.combine_chunks() if isinstance(pool_col, pa.ChunkedArray) else pool_col
    enc = pc.dictionary_encode(pa.concat_arrays([s1_col, pool_col]))
    ids = enc.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    d = enc.dictionary.to_pylist()
    if "" in d:
        ids[ids == d.index("")] = -1
    return ids[:len(s1_col)], ids[len(s1_col):], len(d)


# ========================================================================================
# composite keys (pure numpy; int64 codes)
# ========================================================================================
def _dedup(r, k):
    """unique (r, k) pairs, sorted by r then k"""
    if len(r) == 0:
        return r.astype(np.int64), k.astype(np.int64)
    base = int(k.max()) + 1
    code = np.unique(r.astype(np.int64) * base + k)
    return code // base, code % base


def pairs_within(r, k, V, keep=None):
    """all unordered token pairs inside each record -> code a*V + b (a < b)"""
    if keep is not None:
        r, k = r[keep], k[keep]
    r, k = _dedup(r, k)
    if len(r) == 0:
        return r, k
    starts = np.r_[0, np.nonzero(np.diff(r))[0] + 1]
    sizes = np.diff(np.r_[starts, len(r)])
    pos = np.arange(len(r)) - np.repeat(starts, sizes)
    npart = np.repeat(sizes, sizes) - pos - 1                 # partners after this element
    left = np.repeat(np.arange(len(r)), npart)
    right = left + 1 + (np.arange(int(npart.sum())) - np.repeat(np.cumsum(npart) - npart, npart))
    return r[left], k[left] * V + k[right]


def cross(ra, ka, rb, kb, Vb):
    """every (a, b) token combination of the same record -> code a*Vb + b"""
    lo = np.searchsorted(rb, ra, "left")
    cnt = np.searchsorted(rb, ra, "right") - lo
    left = np.repeat(np.arange(len(ra)), cnt)
    right = np.repeat(lo, cnt) + (np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    return ra[left], ka[left] * Vb + kb[right]


def qualify(r, k, attr, Va):
    """token combined with a record attribute (state / house number) -> code k*Va + attr"""
    a = attr[r]
    ok = a >= 0
    return r[ok], k[ok] * Va + a[ok]


# ========================================================================================
# index
# ========================================================================================
class KeyIndex:
    """Global key space over all families: pool postings (CSR by key) and S1 query postings."""

    def __init__(self, s1, pool, families=FAMILIES, log=print):
        self.n_s1, self.n_pool = len(s1), len(pool)
        t0 = time.time()
        (n1, n2, Vn) = encode_tokens(s1["name_core"], pool["name_core"])
        (p1, p2, Vp) = encode_tokens(s1["name_phon"], pool["name_phon"])
        (a1, a2, Va) = encode_tokens(s1["addr_words"], pool["addr_words"])
        (u1, u2, _) = encode_tokens(s1["addr_nums"], pool["addr_nums"])
        (c1, c2, _) = encode_tokens(s1["name_concat"], pool["name_concat"], explode=False)
        st1, st2, Vs = encode_attr(s1["state_inf"], pool["state_inf"])
        hn1, hn2, Vh = encode_attr(s1["house_no"], pool["house_no"])
        df_a = np.bincount(a2[1], minlength=Va)
        keep1 = df_a[a1[1]] <= ADDR_PAIR_MAX_SHARE * self.n_pool
        keep2 = df_a[a2[1]] <= ADDR_PAIR_MAX_SHARE * self.n_pool
        builders = {   # built one family at a time to keep peak memory low
            "name_rare":    lambda: (n1, n2),
            "name_pair":    lambda: (pairs_within(*n1, Vn), pairs_within(*n2, Vn)),
            "phon_pair":    lambda: (pairs_within(*p1, Vp), pairs_within(*p2, Vp)),
            "name_x_state": lambda: (qualify(*n1, st1, Vs), qualify(*n2, st2, Vs)),
            "name_x_addr":  lambda: (cross(*n1, *a1, Va), cross(*n2, *a2, Va)),
            "concat":       lambda: (c1, c2),
            "hn_x_addr":    lambda: (qualify(*a1, hn1, Vh), qualify(*a2, hn2, Vh)),
            "num_x_addr":   lambda: (cross(*u1, *a1, Va), cross(*u2, *a2, Va)),
            "num_x_state":  lambda: (qualify(*u1, st1, Vs), qualify(*u2, st2, Vs)),
            "addr_pair":    lambda: (pairs_within(*a1, Va, keep1), pairs_within(*a2, Va, keep2)),
            "addr_x_state": lambda: (qualify(*a1, st1, Vs), qualify(*a2, st2, Vs)),
        }
        q_rec, q_key, p_rec, p_key, key_w, key_name = [], [], [], [], [], []
        offset = 0
        for fam, (cap, weight) in families.items():
            (r1, k1), (r2, k2) = builders[fam]()
            u, inv = np.unique(np.concatenate([k1, k2]), return_inverse=True)
            r1d, i1d = _dedup(r1, inv[:len(k1)])
            r2d, i2d = _dedup(r2, inv[len(k1):])
            del r1, k1, r2, k2, inv
            df = np.bincount(i2d, minlength=len(u))
            key_w.append((weight * np.log(self.n_pool / np.maximum(df, 1))).astype(np.float32))
            key_name.append(np.full(len(u), fam in NAME_FAMILIES))
            ok = (df[i1d] >= 1) & (df[i1d] <= cap)
            q_rec.append(r1d[ok].astype(np.int32)); q_key.append((i1d[ok] + offset).astype(np.int32))
            okp = df[i2d] <= cap
            p_rec.append(r2d[okp].astype(np.int32)); p_key.append((i2d[okp] + offset).astype(np.int32))
            offset += len(u)
            del u, r1d, i1d, r2d, i2d, df, ok, okp
            gc.collect()
        del builders, n1, n2, p1, p2, a1, a2, u1, u2, c1, c2
        self.n_keys = offset
        self.key_w = np.concatenate(key_w)
        self.key_name = np.concatenate(key_name)
        q_rec, q_key = np.concatenate(q_rec), np.concatenate(q_key)
        o = np.argsort(q_rec, kind="stable")
        self.q_rec, self.q_key = q_rec[o], q_key[o]
        self.q_ptr = np.searchsorted(self.q_rec, np.arange(self.n_s1 + 1))
        del q_rec, q_key, o
        p_rec, p_key = np.concatenate(p_rec), np.concatenate(p_key)
        # self weight of a record = total weight of its usable keys (for the weighted Jaccard)
        self.s1_self = np.bincount(self.q_rec, weights=self.key_w[self.q_key], minlength=self.n_s1).astype(np.float32)
        self.pool_self = np.bincount(p_rec, weights=self.key_w[p_key], minlength=self.n_pool).astype(np.float32)
        o = np.argsort(p_key, kind="stable")
        self.p_rec = p_rec[o]
        self.p_ptr = np.searchsorted(p_key[o], np.arange(self.n_keys + 1))
        del p_rec, p_key, o
        gc.collect()
        log(f"    index: {self.n_keys:,} keys, {len(self.q_key):,} S1 postings, {len(self.p_rec):,} pool postings "
            f"({time.time() - t0:,.0f}s)")

    def score_range(self, lo, hi):
        """all (s1, pool) pairs sharing >= 1 key, for S1 records lo..hi-1"""
        a, b = self.q_ptr[lo], self.q_ptr[hi]
        rec, key = self.q_rec[a:b].astype(np.int64), self.q_key[a:b]
        starts = self.p_ptr[key]
        lens = self.p_ptr[key + 1] - starts
        tot = int(lens.sum())
        if tot == 0:
            return None
        rep = np.repeat(np.arange(len(rec)), lens)
        pos = np.arange(tot) - np.repeat(np.cumsum(lens) - lens, lens) + starts[rep]
        uniq, inv = np.unique(rec[rep] * self.n_pool + self.p_rec[pos], return_inverse=True)
        w = self.key_w[key]
        score = np.bincount(inv, weights=w[rep]).astype(np.float32)
        name_score = np.bincount(inv, weights=(w * self.key_name[key])[rep]).astype(np.float32)
        nk = np.bincount(inv).astype(np.int32)
        s, p = uniq // self.n_pool, uniq % self.n_pool
        wj = (score / np.maximum(self.s1_self[s] + self.pool_self[p] - score, 1e-6)).astype(np.float32)
        return s, p, score, name_score, wj, nk

    # ---- shared memory: workers read the SAME index bytes (no per-process copy) ----
    ARRAYS = ["q_rec", "q_key", "q_ptr", "p_rec", "p_ptr", "key_w", "key_name", "s1_self", "pool_self"]

    def share(self):
        """move the index arrays into shared memory; returns a small spec for KeyIndex.attach"""
        from multiprocessing import shared_memory
        spec = {"n_s1": self.n_s1, "n_pool": self.n_pool, "n_keys": self.n_keys, "arrays": {}}
        self._shm = []
        for name in self.ARRAYS:
            arr = np.ascontiguousarray(getattr(self, name))
            shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
            view = np.ndarray(arr.shape, arr.dtype, buffer=shm.buf)
            view[...] = arr
            setattr(self, name, view)
            spec["arrays"][name] = (shm.name, arr.shape, arr.dtype.str)
            self._shm.append(shm)
        gc.collect()
        return spec

    @classmethod
    def attach(cls, spec):
        from multiprocessing import shared_memory
        obj = cls.__new__(cls)
        obj.n_s1, obj.n_pool, obj.n_keys = spec["n_s1"], spec["n_pool"], spec["n_keys"]
        obj._shm = []
        for name, (shm_name, shape, dtype) in spec["arrays"].items():
            shm = shared_memory.SharedMemory(name=shm_name)
            setattr(obj, name, np.ndarray(shape, np.dtype(dtype), buffer=shm.buf))
            obj._shm.append(shm)
        return obj

    def release(self):
        for name in self.ARRAYS:
            setattr(self, name, None)
        for shm in getattr(self, "_shm", []):
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        self._shm = []


# ========================================================================================
# candidate lists + re-ranking features
# ========================================================================================
def _group_bounds(s):
    starts = np.r_[0, np.nonzero(np.diff(s))[0] + 1]
    return starts, np.diff(np.r_[starts, len(s)])


def rank_within(s, value):
    """rank (0 = highest value) inside each S1 group; s must be sorted"""
    o = np.lexsort((-value, s))
    starts, sizes = _group_bounds(s[o])
    r = np.arange(len(s)) - np.repeat(starts, sizes)
    out = np.empty(len(s), np.int32)
    out[o] = r
    return out


def generate_chunk(index, lo, hi, k_wj, k_score, k_name):
    """union of the three short lists for S1 records lo..hi-1 -> dict of columns (sorted by s)"""
    res = index.score_range(lo, hi)
    if res is None:
        return None
    s, p, score, name_score, wj, nk = res           # s sorted (from np.unique on s*N+p)
    r_sc = rank_within(s, score)
    r_wj = rank_within(s, wj)
    has_name = name_score > 0
    r_nm = rank_within(s, np.where(has_name, name_score, -1.0).astype(np.float32))
    keep = (r_wj < k_wj) | (r_sc < k_score) | ((r_nm < k_name) & has_name)
    d = {"s": s[keep], "p": p[keep], "score": score[keep], "wj": wj[keep], "name_score": name_score[keep],
         "nk": nk[keep], "r_score": r_sc[keep], "r_wj": r_wj[keep], "r_name": r_nm[keep]}
    # features relative to the entity's best candidate
    starts, sizes = _group_bounds(d["s"])
    for c in ["score", "wj", "name_score"]:
        mx = np.repeat(np.maximum.reduceat(d[c], starts), sizes)
        d[c + "_rel"] = d[c] / np.maximum(mx, 1e-6)
        d[c + "_gap"] = mx - d[c]
    d["n_cand"] = np.repeat(sizes, sizes).astype(np.int32)
    d["name_share"] = d["name_score"] / np.maximum(d["score"], 1e-6)
    return d


def feature_matrix(d):
    return np.column_stack([d[f].astype(np.float32) for f in RERANK_FEATS])


def candidates_for(index, blk):
    """candidate dict (or None) for a sorted block of S1 record ids"""
    br = np.r_[0, np.nonzero(np.diff(blk) != 1)[0] + 1, len(blk)]
    parts = [generate_chunk(index, blk[x], blk[y - 1] + 1, **LISTS) for x, y in zip(br[:-1], br[1:])]
    parts = [q for q in parts if q is not None]
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]} if parts else None


def recompute_group_features(d):
    """Re-sort by s and recompute all relative, gap, and within-group rank features after a merge."""
    o = np.argsort(d["s"], kind="stable")
    for key in list(d.keys()):
        d[key] = d[key][o]
    starts, sizes = _group_bounds(d["s"])
    for c in ["score", "wj", "name_score"]:
        mx = np.repeat(np.maximum.reduceat(d[c], starts), sizes)
        d[c + "_rel"] = (d[c] / np.maximum(mx, 1e-6)).astype(np.float32)
        d[c + "_gap"] = (mx - d[c]).astype(np.float32)
    d["n_cand"] = np.repeat(sizes, sizes).astype(np.int32)
    d["name_share"] = (d["name_score"] / np.maximum(d["score"], 1e-6)).astype(np.float32)
    d["r_score"] = rank_within(d["s"], d["score"])
    d["r_wj"] = rank_within(d["s"], d["wj"])
    has_name = d["name_score"] > 0
    d["r_name"] = rank_within(d["s"], np.where(has_name, d["name_score"], -1.0).astype(np.float32))
    return d


def merge_tfidf(d, blk, tfidf_extra):
    """Inject TF-IDF candidates absent from d. Returns updated dict (or d unchanged if no extras)."""
    if not tfidf_extra:
        return d
    existing = set()
    if d is not None:
        existing = set(zip(d["s"].tolist(), d["p"].tolist()))
    new_s, new_p = [], []
    for s_idx in blk:
        extras = tfidf_extra.get(int(s_idx))
        if extras is not None:
            for p_idx in extras:
                pair = (int(s_idx), int(p_idx))
                if pair not in existing:
                    new_s.append(s_idx)
                    new_p.append(p_idx)
                    existing.add(pair)
    if not new_s:
        return d
    n = len(new_s)
    z_f, z_i = np.zeros(n, np.float32), np.zeros(n, np.int32)
    new_d = {
        "s": np.array(new_s, np.int64), "p": np.array(new_p, np.int64),
        "score": z_f.copy(), "wj": z_f.copy(), "name_score": z_f.copy(), "nk": z_i.copy(),
        "r_score": z_i.copy(), "r_wj": z_i.copy(), "r_name": z_i.copy(),
        "score_rel": z_f.copy(), "wj_rel": z_f.copy(), "name_score_rel": z_f.copy(),
        "score_gap": z_f.copy(), "wj_gap": z_f.copy(), "name_score_gap": z_f.copy(),
        "n_cand": z_i.copy(), "name_share": z_f.copy(),
    }
    merged = new_d if d is None else {k: np.concatenate([d[k], new_d[k]]) for k in d}
    return recompute_group_features(merged)


def process_block(index, blk, mode, models=None, s1_fold=None, k=None, tfidf_extra=None):
    """The unit of work (identical in serial and parallel runs).
    mode 'sample' -> (s, p, feature matrix) of every generated candidate
    mode 'full'   -> re-ranked top-k candidates of every S1 record in blk"""
    d = candidates_for(index, blk)
    if tfidf_extra:
        d = merge_tfidf(d, blk, tfidf_extra)
    if mode == "sample":
        return None if d is None else (d["s"], d["p"], feature_matrix(d))
    if d is None:
        return None
    X = feature_matrix(d)
    prob = np.empty(len(X), np.float32)
    fo = s1_fold[d["s"]]
    for f in np.unique(fo):
        m = fo == f
        prob[m] = models[int(f) if int(f) in models else -1].predict_proba(X[m])[:, 1]
    rr = rank_within(d["s"], prob)
    keep = rr < k
    o = np.lexsort((rr[keep], d["s"][keep]))
    out = {c: d[c][keep][o] for c in ["s", "p", "score", "wj", "name_score", "nk"]}
    out.update({"prob": prob[keep][o], "rr": rr[keep][o], "fold": fo[keep][o]})
    return out


_W = {}


def _init_worker(spec, mode, models, s1_fold, k, tfidf_extra):
    _W.update(index=KeyIndex.attach(spec), mode=mode, models=models,
              s1_fold=s1_fold, k=k, tfidf_extra=tfidf_extra)


def _work(blk):
    return blk, process_block(_W["index"], blk, _W["mode"], _W["models"],
                               _W["s1_fold"], _W["k"], _W["tfidf_extra"])


GB_PER_WORKER = 0.8        # working memory of one worker for a 2000-record chunk (measured peak + margin)


def auto_workers(log=print):
    """as many workers as the machine can hold: min(cores - 1, (free RAM - 1 GB) / GB_PER_WORKER)"""
    import psutil
    cores = os.cpu_count() or 2
    free_gb = psutil.virtual_memory().available / 1e9
    n = max(1, min(cores - 1, int((free_gb - 1.0) / GB_PER_WORKER)))
    log(f"    auto workers: {n} (cores {cores}, free RAM {free_gb:.1f} GB)")
    return n


def tfidf_block(s1, pool, top_k_name=15, top_k_addr=8, ngram=(2, 4), log=print):
    """Char n-gram TF-IDF ANN blocking.
    Returns dict {s1_idx (int): np.array(pool_indices, int32)}.
    Runs name blocking always; address blocking only for records with >=3 addr words."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    s1_name = to_np(s1["name_core"]).tolist()
    pool_name = to_np(pool["name_core"]).tolist()
    s1_addr = to_np(s1["addr_words"]).tolist()
    pool_addr = to_np(pool["addr_words"]).tolist()
    result = {}

    for label, texts_s1, texts_pool, top_k, min_words in [
        ("name", s1_name, pool_name, top_k_name, 0),
        ("addr", s1_addr, pool_addr, top_k_addr, 3),
    ]:
        if top_k == 0:
            continue
        active = np.array([i for i, t in enumerate(texts_s1)
                           if len(t.split()) >= min_words], dtype=np.int32)
        if len(active) == 0:
            continue
        s1_active = [texts_s1[i] if texts_s1[i] else "_" for i in active]
        pool_clean = [t if t else "_" for t in texts_pool]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, max_df=0.4,
                              sublinear_tf=True, dtype=np.float32, min_df=2)
        try:
            pool_mat = vec.fit_transform(pool_clean)
            s1_mat = vec.transform(s1_active)
        except ValueError:
            continue
        k_actual = min(top_k, pool_mat.shape[0])
        chunk = 200
        for lo in range(0, len(active), chunk):
            hi = min(lo + chunk, len(active))
            sims = (s1_mat[lo:hi] @ pool_mat.T).toarray()
            for i, row in enumerate(sims):
                top = np.argpartition(row, -k_actual)[-k_actual:]
                top = top[row[top] > 0.05]
                if len(top) > 0:
                    s_idx = int(active[lo + i])
                    result.setdefault(s_idx, set()).update(int(j) for j in top)
        log(f"    tfidf {label}: {len(active):,} active S1, "
            f"{sum(len(v) for v in result.values()):,} cumulative candidate pairs")

    return {k: np.array(sorted(v), dtype=np.int32) for k, v in result.items()}


def run_blocks(index, ids, chunk, workers, mode, models=None, s1_fold=None, k=None, tfidf_extra=None):
    """yield (blk, result) in S1 order, serially or with a pool of workers sharing the index"""
    blocks = [ids[i:i + chunk] for i in range(0, len(ids), chunk)]
    if workers == 0:
        workers = auto_workers()
    if workers <= 1:
        for blk in blocks:
            yield blk, process_block(index, blk, mode, models, s1_fold, k, tfidf_extra)
        return
    from multiprocessing import Pool
    spec = index.share()
    try:
        with Pool(workers, initializer=_init_worker,
                  initargs=(spec, mode, models, s1_fold, k, tfidf_extra)) as pool:
            for blk, res in pool.imap(_work, blocks, chunksize=1):
                yield blk, res
    finally:
        index.release()


# ========================================================================================
# pipeline
# ========================================================================================
def load_split(split, clean_dir):
    s1 = load_clean(split, 1, clean_dir)
    pool = pa.concat_tables([load_clean(split, 2, clean_dir), load_clean(split, 3, clean_dir)]).combine_chunks()
    return s1, pool


def country_tables(s1, pool, country):
    return s1.filter(pc.equal(s1["country"], country)), pool.filter(pc.equal(pool["country"], country))


def true_links(data):
    gt = read_tsv(os.path.join(data, "train", "train_ground_truth.tsv"))
    g = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    g = g[g.m != ""]
    return dict(zip(g.m, g.source1_entity_id))


def train_rerankers(a, log):
    """Phase 1: generate candidates for a sample of train entities, fit one model per held-out fold
    (cross-fitting) plus one model on all folds (for test)."""
    folds = read_tsv(a.folds)
    fold_of = dict(zip(folds.entity_id, folds.fold.astype(int)))
    link = true_links(a.data)
    s1_all, pool_all = load_split("train", a.clean)
    rng = np.random.default_rng(a.seed)
    X, Y, F = [], [], []
    for country in sorted(set(to_np(s1_all["country"]))):
        s1, pool = country_tables(s1_all, pool_all, country)
        s1_ids, pool_ids = to_np(s1["entity_id"]), to_np(pool["entity_id"])
        log(f"  [rerank sample] train {country}: S1 {len(s1_ids):,}, pool {len(pool_ids):,}")
        index = KeyIndex(s1, pool, log=log)
        tfidf_extra = None
        if getattr(a, "tfidf_topk_name", 0) > 0 or getattr(a, "tfidf_topk_addr", 0) > 0:
            log(f"    [{country}] tfidf blocking (name top {a.tfidf_topk_name}, addr top {a.tfidf_topk_addr})")
            tfidf_extra = tfidf_block(s1, pool, a.tfidf_topk_name, a.tfidf_topk_addr, log=log)
        sample = np.sort(rng.choice(len(s1_ids), size=int(len(s1_ids) * a.sample_frac), replace=False))
        s1_pos = {e: i for i, e in enumerate(s1_ids)}
        n_pool = len(pool_ids)
        truth = np.sort(np.array([s1_pos[link[e]] * n_pool + j for j, e in enumerate(pool_ids)
                                  if e in link and link[e] in s1_pos], np.int64))
        s1_fold = np.array([fold_of[e] for e in s1_ids], np.int8)
        for _, res in run_blocks(index, sample, a.chunk, a.workers, "sample", tfidf_extra=tfidf_extra):
            if res is None:
                continue
            s, p, x = res
            X.append(x)
            Y.append(np.isin(s * n_pool + p, truth, assume_unique=False))
            F.append(s1_fold[s])
        del index, s1, pool, truth
        gc.collect()
    X, Y, F = np.concatenate(X), np.concatenate(Y), np.concatenate(F)
    log(f"  [rerank] training pairs {len(Y):,} (positives {Y.mean():.3%})")
    models = {}
    for f in sorted(set(F.tolist())) + [-1]:
        m = F != f if f >= 0 else np.ones(len(F), bool)
        t0 = time.time()
        models[f] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=63,
                                                   random_state=0).fit(X[m], Y[m])
        log(f"  [rerank] model for fold {f if f >= 0 else 'ALL (test)'} trained on {m.sum():,} pairs "
            f"({time.time() - t0:.0f}s)")
    del s1_all, pool_all, X, Y, F
    gc.collect()
    return models


def block_split(split, a, models, log):
    """Phase 2: candidates for every Source-1 record of a split -> TSV"""
    fold_of = {}
    if split == "train":
        folds = read_tsv(a.folds)
        fold_of = dict(zip(folds.entity_id, folds.fold.astype(int)))
    s1_all, pool_all = load_split(split, a.clean)
    dst = os.path.join(a.out, f"{split}_candidates.tsv")
    grouped_dst = os.path.join(a.out, "candidate_pairs.tsv") if split == "test" else None
    header = True
    n_pairs = n_s1_rows = 0
    for country in sorted(set(to_np(s1_all["country"]))):
        s1, pool = country_tables(s1_all, pool_all, country)
        s1_ids, pool_ids = to_np(s1["entity_id"]), to_np(pool["entity_id"])
        log(f"  [{split} {country}] S1 {len(s1_ids):,}, pool {len(pool_ids):,}")
        index = KeyIndex(s1, pool, log=log)
        tfidf_extra = None
        if getattr(a, "tfidf_topk_name", 0) > 0 or getattr(a, "tfidf_topk_addr", 0) > 0:
            log(f"  [{split} {country}] tfidf blocking (name top {a.tfidf_topk_name}, addr top {a.tfidf_topk_addr})")
            tfidf_extra = tfidf_block(s1, pool, a.tfidf_topk_name, a.tfidf_topk_addr, log=log)
        t0 = time.time()
        s1_fold = np.array([fold_of.get(e, -1) for e in s1_ids], np.int8)
        blocks = run_blocks(index, np.arange(len(s1_ids)), a.chunk, a.workers, "full",
                            models, s1_fold, a.k, tfidf_extra=tfidf_extra)
        for ci, (blk, r) in enumerate(blocks):
            if r is None:          # no candidate at all for the whole block
                r = {c: np.empty(0, np.int64) for c in ["s", "p", "nk", "rr", "fold"]}
                r.update({c: np.empty(0, np.float32) for c in ["score", "wj", "name_score", "prob"]})
            out = pd.DataFrame({
                "source1_entity_id": s1_ids[r["s"]], "candidate_entity_id": pool_ids[r["p"]],
                "country": country, "fold": r["fold"], "blk_score": np.round(r["score"].astype(np.float64), 3),
                "blk_wj": np.round(r["wj"].astype(np.float64), 5),
                "blk_name_score": np.round(r["name_score"].astype(np.float64), 3),
                "blk_nkeys": r["nk"], "rr_prob": np.round(r["prob"].astype(np.float64), 6), "rr_rank": r["rr"]})
            out[OUT_COLS].to_csv(dst, sep="\t", index=False, header=header, mode="w" if header else "a",
                                 quoting=csv.QUOTE_NONE, lineterminator="\n")
            if grouped_dst:        # submission format: one row per Source-1 entity, empty list allowed
                lists = out.groupby("source1_entity_id", sort=False).candidate_entity_id.agg(",".join)
                g = pd.DataFrame({"source1_entity_id": s1_ids[blk]})
                g["candidate_entity_ids"] = g.source1_entity_id.map(lists).fillna("")
                g.to_csv(grouped_dst, sep="\t", index=False, header=header, mode="w" if header else "a",
                         quoting=csv.QUOTE_NONE, lineterminator="\n")
                n_s1_rows += len(g)
            header = False
            n_pairs += len(out)
            if ci % 50 == 0:
                done = min((ci + 1) * a.chunk, len(s1_ids))
                log(f"    {done:,}/{len(s1_ids):,} S1 done ({time.time() - t0:,.0f}s)")
        log(f"  [{split} {country}] done in {time.time() - t0:,.0f}s")
        del index, s1, pool
        gc.collect()
    log(f"  [{split}] {n_pairs:,} candidate pairs -> {dst}")
    if grouped_dst:
        log(f"  [{split}] submission-format candidate list: {n_s1_rows:,} Source-1 rows -> {grouped_dst}")


def evaluate_train(a, log):
    """recall and F0.5 ceiling of the (out-of-fold) train candidates against the ground truth.
    Uses an Arrow join so the ~44M candidate rows never become Python strings."""
    gt = read_tsv(os.path.join(a.data, "train", "train_ground_truth.tsv"))
    # evaluate exactly the Source-1 entities that were blocked (the cleaned train Source 1)
    s1 = load_clean("train", 1, a.clean, columns=["entity_id", "country"]).to_pandas()
    folds = read_tsv(a.folds)
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    ex = ex[ex.m != ""]
    truth = pa.table({"source1_entity_id": pa.array(ex.source1_entity_id.tolist(), type=pa.string()),
                      "candidate_entity_id": pa.array(ex.m.tolist(), type=pa.string())})
    cand = pacsv.read_csv(
        os.path.join(a.out, "train_candidates.tsv"),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(include_columns=["source1_entity_id", "candidate_entity_id", "rr_rank"],
                                             column_types={"source1_entity_id": pa.string(),
                                                           "candidate_entity_id": pa.string(), "rr_rank": pa.int32()}))
    found = truth.join(cand, ["source1_entity_id", "candidate_entity_id"], join_type="left outer").to_pandas()
    found["rr_rank"] = found.rr_rank.fillna(10 ** 6)
    n_true = gt.set_index("source1_entity_id").matched_entity_ids.map(lambda v: len([x for x in v.split(",") if x]))
    ent = s1.merge(folds.rename(columns={"entity_id": "entity_id"}), on="entity_id")
    ent["n_true"] = ent.entity_id.map(n_true).values
    lines = [f"Blocking evaluation on TRAIN (out-of-fold re-ranking) vs train_ground_truth.tsv; "
             f"{len(truth):,} true pairs, {len(cand):,} candidate pairs"]
    for K in sorted({5, 10, 15, a.k}):
        hit = found[found.rr_rank < K].groupby("source1_entity_id").size()
        ent["found"] = ent.entity_id.map(hit).fillna(0).values
        r = ent.found / ent.n_true.where(ent.n_true > 0, 1)
        ent["f05"] = np.where(ent.n_true == 0, 1.0, np.where(r > 0, 1.25 * r / (0.25 + r), 0.0))
        per = " | ".join(f"{ct}: recall {h.found.sum() / h.n_true.sum():.4f}, ceiling {h.f05.mean():.4f}"
                         for ct, h in ent.groupby("country"))
        lines.append(f"K={K:2d}: pair recall {ent.found.sum() / ent.n_true.sum():.4f} | "
                     f"F0.5 ceiling {ent.f05.mean():.4f} | {per}")
        if K == a.k:
            lines.append(f"      per fold: " + " | ".join(f"fold {f}: recall {h.found.sum() / h.n_true.sum():.4f}"
                                                         for f, h in ent.groupby("fold")))
    for l in lines:
        log(l)
    open(os.path.join(a.out, "blocking_report.txt"), "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Blocking / candidate generation")
    ap.add_argument("--data", required=True, help="raw dataset folder (ground truth for train)")
    ap.add_argument("--clean", default=CLEAN, help="cleaned dataset folder from data_cleaning.py")
    ap.add_argument("--folds", default=os.path.join(ROOT, "artifacts", "folds.tsv"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--k", type=int, default=20, help="candidates kept per Source-1 record")
    ap.add_argument("--chunk", type=int, default=2000, help="S1 records per work unit")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel processes sharing one index in shared memory: 0 = auto (from cores and "
                         "free RAM), 1 = serial, N = exactly N.  Results are identical for any value")
    ap.add_argument("--sample_frac", type=float, default=0.10, help="share of train entities used to fit the re-ranker")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--tfidf_topk_name", type=int, default=15,
                    help="char n-gram TF-IDF name ANN top-k per S1 record (0 = disable)")
    ap.add_argument("--tfidf_topk_addr", type=int, default=8,
                    help="char n-gram TF-IDF address ANN top-k per S1 record (0 = disable)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "blocking_log.txt"), "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    model_path = os.path.join(a.out, "rerankers.pkl")
    if os.path.exists(model_path):
        models = pickle.load(open(model_path, "rb"))
        log(f"loaded re-rankers from {model_path}")
    else:
        log("phase 1: fitting the cross-fitted re-rankers")
        models = train_rerankers(a, log)
        pickle.dump(models, open(model_path, "wb"))
    for split in a.splits.split(","):
        log(f"phase 2: blocking {split}")
        block_split(split, a, models, log)
        if split == "train":
            evaluate_train(a, log)


if __name__ == "__main__":
    main()
