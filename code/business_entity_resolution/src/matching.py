"""Matching model -- single entry point for Step 4.

Run from code/business_entity_resolution/ (after blocking.py):

    python src/matching.py --data ../../dataset --phase features     # pair features (train + test)
    python src/matching.py --data ../../dataset --phase train        # OOF model + test model

Every candidate pair produced by blocking gets ~50 language-neutral features (country is NOT a
feature, so the model transfers to France):
  * blocking signals      rarity score, weighted Jaccard, name-only score, shared keys, re-rank prob/rank
  * name similarity       fuzzy ratios on the cleaned name, phonetic name, glued name (domains),
                          acronym match, legal-form agreement, record type, native-script flag
  * address similarity    word / number ratios, house number & unit equal/different/missing,
                          city, state, empty-address flags
  * competition           how the pair compares with the entity's best candidate and with the
                          other entities claiming the same Source-2/3 record
Features are cached as float32 .npy files aligned row-by-row with blocking's *_candidates.tsv.
"""
import argparse
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLEAN = os.path.join(ROOT, "artifacts", "cleaned_dataset")
BLOCK = os.path.join(ROOT, "artifacts", "blocking")
FEAT = os.path.join(ROOT, "artifacts", "features")

CLEAN_COLS = ["entity_id", "name_core", "name_legal", "name_brand", "name_type", "name_script", "name_concat",
              "name_acronym", "name_phon", "addr_words", "addr_nums", "house_no", "unit", "city", "state_inf",
              "addr_missing"]
NAME_TYPES = {"normal": 0, "domain": 1, "handle": 2, "brand": 3, "alias": 4, "acronym": 5}

FEATURES = [
    # blocking
    "blk_score", "blk_wj", "blk_name_score", "blk_nkeys", "rr_prob", "rr_rank",
    # name
    "n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw", "p_ratio", "p_tset", "c_lev", "c_partial", "c_eq",
    "acr_match", "legal_eq", "legal_s1_empty", "legal_c_empty", "c_type", "c_native", "c_has_brand",
    "n_tok_s1", "n_tok_c", "n_len_s1", "n_len_c",
    # address
    "a_ratio", "a_tset", "a_tsort", "a_partial", "u_tset", "u_ratio", "house_eq", "house_lev", "unit_eq",
    "city_eq", "city_jw", "state_eq", "addr_empty_c", "a_tok_s1", "a_tok_c", "u_tok_s1", "u_tok_c",
    # competition (computed over the whole candidate set of the split)
    "s1_n_cand", "s1_max_rr", "rr_rel_s1", "c_n_s1", "c_max_rr", "rr_rel_c", "c_rank", "c_gap",
]


# ========================================================================================
# loading
# ========================================================================================
def read_arrow(path, types):
    return pacsv.read_csv(
        path, parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(column_types=types, include_columns=list(types),
                                             strings_can_be_null=False, quoted_strings_can_be_null=False),
        read_options=pacsv.ReadOptions(block_size=1 << 26)).combine_chunks()


def load_clean_tables(split, clean_dir=CLEAN):
    types = {c: pa.string() for c in CLEAN_COLS}
    s1 = read_arrow(os.path.join(clean_dir, split, f"{split}_source1.tsv"), types)
    pool = pa.concat_tables([read_arrow(os.path.join(clean_dir, split, f"{split}_source{k}.tsv"), types)
                             for k in "23"]).combine_chunks()
    return s1, pool


def load_candidates(split, block_dir=BLOCK):
    types = {"source1_entity_id": pa.string(), "candidate_entity_id": pa.string(), "fold": pa.int8(),
             "blk_score": pa.float32(), "blk_wj": pa.float32(), "blk_name_score": pa.float32(),
             "blk_nkeys": pa.float32(), "rr_prob": pa.float32(), "rr_rank": pa.float32()}
    return read_arrow(os.path.join(block_dir, f"{split}_candidates.tsv"), types)


# ========================================================================================
# per-pair features
# ========================================================================================
def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _tri_state(a, b):
    """1 = equal, 0 = different, -1 = at least one side missing"""
    both = pc.and_(pc.not_equal(a, ""), pc.not_equal(b, ""))
    eq = pc.equal(a, b)
    return np.where(both.to_numpy(zero_copy_only=False), eq.to_numpy(zero_copy_only=False).astype(np.float32), -1.0)


def _ntok(col):
    """number of space-separated tokens (0 for empty)"""
    n = pc.add(pc.count_substring(col, " "), 1).to_numpy(zero_copy_only=False).astype(np.float32)
    return np.where(pc.equal(col, "").to_numpy(zero_copy_only=False), 0.0, n)


def pair_features(s1, pool, i1, i2, cand_chunk):
    """features of one chunk of candidate pairs (i1, i2 = row indices into s1 / pool)"""
    A = {c: pc.take(s1[c], pa.array(i1)) for c in CLEAN_COLS[1:]}
    B = {c: pc.take(pool[c], pa.array(i2)) for c in CLEAN_COLS[1:]}
    L = {k: v.to_pylist() for k, v in A.items() if k in ("name_core", "name_phon", "name_concat", "addr_words",
                                                            "addr_nums", "house_no", "city")}
    R = {k: v.to_pylist() for k, v in B.items() if k in L}
    f = {}
    for c in ["blk_score", "blk_wj", "blk_name_score", "blk_nkeys", "rr_prob", "rr_rank"]:
        f[c] = cand_chunk[c].to_numpy(zero_copy_only=False).astype(np.float32)
    n1, n2 = L["name_core"], R["name_core"]
    f["n_ratio"] = _cp(n1, n2, fuzz.ratio)
    f["n_tset"] = _cp(n1, n2, fuzz.token_set_ratio)
    f["n_tsort"] = _cp(n1, n2, fuzz.token_sort_ratio)
    f["n_partial"] = _cp(n1, n2, fuzz.partial_ratio)
    f["n_jw"] = _cp(n1, n2, JaroWinkler.normalized_similarity)
    f["p_ratio"] = _cp(L["name_phon"], R["name_phon"], fuzz.ratio)
    f["p_tset"] = _cp(L["name_phon"], R["name_phon"], fuzz.token_set_ratio)
    f["c_lev"] = _cp(L["name_concat"], R["name_concat"], Levenshtein.normalized_similarity)
    f["c_partial"] = _cp(L["name_concat"], R["name_concat"], fuzz.partial_ratio)
    f["c_eq"] = _tri_state(A["name_concat"], B["name_concat"])
    acr = pc.or_(pc.and_(pc.not_equal(A["name_acronym"], ""), pc.equal(A["name_acronym"], B["name_concat"])),
                 pc.and_(pc.not_equal(B["name_acronym"], ""), pc.equal(B["name_acronym"], A["name_concat"])))
    f["acr_match"] = acr.to_numpy(zero_copy_only=False).astype(np.float32)
    f["legal_eq"] = _tri_state(A["name_legal"], B["name_legal"])
    f["legal_s1_empty"] = pc.equal(A["name_legal"], "").to_numpy(zero_copy_only=False).astype(np.float32)
    f["legal_c_empty"] = pc.equal(B["name_legal"], "").to_numpy(zero_copy_only=False).astype(np.float32)
    f["c_type"] = np.array([NAME_TYPES.get(t, 0) for t in B["name_type"].to_pylist()], np.float32)
    f["c_native"] = pc.not_equal(B["name_script"], "latin").to_numpy(zero_copy_only=False).astype(np.float32)
    f["c_has_brand"] = pc.not_equal(B["name_brand"], "").to_numpy(zero_copy_only=False).astype(np.float32)
    f["n_tok_s1"], f["n_tok_c"] = _ntok(A["name_core"]), _ntok(B["name_core"])
    f["n_len_s1"] = pc.utf8_length(A["name_core"]).to_numpy(zero_copy_only=False).astype(np.float32)
    f["n_len_c"] = pc.utf8_length(B["name_core"]).to_numpy(zero_copy_only=False).astype(np.float32)
    a1, a2 = L["addr_words"], R["addr_words"]
    f["a_ratio"] = _cp(a1, a2, fuzz.ratio)
    f["a_tset"] = _cp(a1, a2, fuzz.token_set_ratio)
    f["a_tsort"] = _cp(a1, a2, fuzz.token_sort_ratio)
    f["a_partial"] = _cp(a1, a2, fuzz.partial_ratio)
    f["u_tset"] = _cp(L["addr_nums"], R["addr_nums"], fuzz.token_set_ratio)
    f["u_ratio"] = _cp(L["addr_nums"], R["addr_nums"], fuzz.ratio)
    f["house_eq"] = _tri_state(A["house_no"], B["house_no"])
    f["house_lev"] = _cp(L["house_no"], R["house_no"], Levenshtein.normalized_similarity)
    f["unit_eq"] = _tri_state(A["unit"], B["unit"])
    f["city_eq"] = _tri_state(A["city"], B["city"])
    f["city_jw"] = _cp(L["city"], R["city"], JaroWinkler.normalized_similarity)
    f["state_eq"] = _tri_state(A["state_inf"], B["state_inf"])
    f["addr_empty_c"] = pc.equal(B["addr_missing"], "1").to_numpy(zero_copy_only=False).astype(np.float32)
    f["a_tok_s1"], f["a_tok_c"] = _ntok(A["addr_words"]), _ntok(B["addr_words"])
    f["u_tok_s1"], f["u_tok_c"] = _ntok(A["addr_nums"]), _ntok(B["addr_nums"])
    # empty strings: similarity functions return 0 or 100 on empty input -> mark as missing (-1)
    for c, src in {"a_ratio": "addr_words", "a_tset": "addr_words", "a_tsort": "addr_words",
                   "a_partial": "addr_words", "u_tset": "addr_nums", "u_ratio": "addr_nums",
                   "house_lev": "house_no", "city_jw": "city", "p_ratio": "name_phon", "p_tset": "name_phon",
                   "c_lev": "name_concat", "c_partial": "name_concat"}.items():
        miss = pc.or_(pc.equal(A[src], ""), pc.equal(B[src], "")).to_numpy(zero_copy_only=False)
        f[c][miss] = -1.0
    return f


# ========================================================================================
# competition features (over the whole candidate set)
# ========================================================================================
def _group_stats(key, value):
    """per-group count, max, rank of value (0 = best), gap to the group's best"""
    o = np.lexsort((-value, key))
    k = key[o]
    starts = np.r_[0, np.nonzero(np.diff(k))[0] + 1]
    sizes = np.diff(np.r_[starts, len(k)])
    mx = np.maximum.reduceat(value[o], starts)
    rank = np.empty(len(key), np.float32)
    rank[o] = np.arange(len(k)) - np.repeat(starts, sizes)
    cnt = np.empty(len(key), np.float32)
    cnt[o] = np.repeat(sizes, sizes)
    gmax = np.empty(len(key), np.float32)
    gmax[o] = np.repeat(mx, sizes)
    return cnt, gmax, rank


def competition_features(i1, i2, rr):
    f = {}
    cnt, mx, _ = _group_stats(i1, rr)
    f["s1_n_cand"], f["s1_max_rr"], f["rr_rel_s1"] = cnt, mx, rr / np.maximum(mx, 1e-6)
    cnt, mx, rank = _group_stats(i2, rr)
    f["c_n_s1"], f["c_max_rr"], f["rr_rel_c"], f["c_rank"] = cnt, mx, rr / np.maximum(mx, 1e-6), rank
    f["c_gap"] = mx - rr
    return f


# ========================================================================================
# phase: features
# ========================================================================================
def build_features(split, a, log):
    t0 = time.time()
    s1, pool = load_clean_tables(split, a.clean)
    cand = load_candidates(split, a.blocks)
    n = len(cand)
    i1 = pc.index_in(cand["source1_entity_id"], value_set=s1["entity_id"]).to_numpy(zero_copy_only=False)
    i2 = pc.index_in(cand["candidate_entity_id"], value_set=pool["entity_id"]).to_numpy(zero_copy_only=False)
    assert not np.isnan(i1.astype(float)).any() and not np.isnan(i2.astype(float)).any(), "unknown ids"
    i1, i2 = i1.astype(np.int64), i2.astype(np.int64)
    log(f"  [{split}] {n:,} pairs, S1 {len(s1):,}, pool {len(pool):,} (loaded in {time.time() - t0:.0f}s)")
    os.makedirs(a.feat, exist_ok=True)
    X = np.lib.format.open_memmap(os.path.join(a.feat, f"{split}_X.npy"), mode="w+", dtype=np.float32,
                                  shape=(n, len(FEATURES)))
    col = {c: j for j, c in enumerate(FEATURES)}
    for lo in range(0, n, a.chunk):
        hi = min(lo + a.chunk, n)
        f = pair_features(s1, pool, i1[lo:hi], i2[lo:hi], cand.slice(lo, hi - lo))
        for c, v in f.items():
            X[lo:hi, col[c]] = v
        log(f"    {hi:,}/{n:,} pairs ({time.time() - t0:,.0f}s)")
    rr = cand["rr_prob"].to_numpy(zero_copy_only=False).astype(np.float32)
    for c, v in competition_features(i1, i2, rr).items():
        X[:, col[c]] = v
    X.flush()
    np.save(os.path.join(a.feat, f"{split}_fold.npy"), cand["fold"].to_numpy(zero_copy_only=False).astype(np.int8))
    if split == "train":
        np.save(os.path.join(a.feat, "train_y.npy"), labels(cand, a.data))
    with open(os.path.join(a.feat, "feature_names.txt"), "w") as fh:
        fh.write("\n".join(FEATURES) + "\n")
    log(f"  [{split}] features {X.shape} -> {a.feat} ({time.time() - t0:,.0f}s)")


def labels(cand, data):
    """1 if the candidate pair is in the training ground truth"""
    gt = read_arrow(os.path.join(data, "train", "train_ground_truth.tsv"),
                    {"source1_entity_id": pa.string(), "matched_entity_ids": pa.string()})
    lists = pc.split_pattern(gt["matched_entity_ids"], ",")
    m = pc.list_flatten(lists)
    s = pc.take(gt["source1_entity_id"], pc.list_parent_indices(lists))
    keep = pc.not_equal(m, "")
    truth_key = pc.binary_join_element_wise(pc.filter(s, keep), pc.filter(m, keep), "|")
    cand_key = pc.binary_join_element_wise(cand["source1_entity_id"], cand["candidate_entity_id"], "|")
    return pc.is_in(cand_key, value_set=truth_key).to_numpy(zero_copy_only=False).astype(np.int8)


# ========================================================================================
# phase: train (out-of-fold on train, final model for test)
# ========================================================================================
XGB_PARAMS = {"objective": "binary:logistic", "eval_metric": ["logloss", "aucpr"], "tree_method": "hist",
              "max_depth": 8, "eta": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
              "lambda": 1.0, "max_bin": 256}


def entity_groups(split, a):
    """group id (Source-1 entity) of every candidate row; rows of an entity are contiguous"""
    s1 = read_arrow(os.path.join(a.blocks, f"{split}_candidates.tsv"), {"source1_entity_id": pa.string()})
    col = s1["source1_entity_id"]
    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    return pc.dictionary_encode(col).indices.to_numpy(zero_copy_only=False).astype(np.int64)


def sample_rows(groups, row_mask, frac, rng):
    """all rows of a random `frac` of the entities among rows where row_mask is True"""
    ents = np.unique(groups[row_mask])
    chosen = np.zeros(groups.max() + 1, bool)
    chosen[rng.choice(ents, size=max(1, int(len(ents) * frac)), replace=False)] = True
    return np.nonzero(row_mask & chosen[groups])[0]


class Rows:
    """row access to one or several aligned feature matrices (memory-mapped), column-concatenated"""

    def __init__(self, *mats):
        self.mats = mats

    def __getitem__(self, idx):
        """rows `idx` of all matrices side by side; filled chunk by chunk into one pre-allocated array so
        the peak memory is ~ the size of the result (no temporary copies of whole blocks)"""
        if len(self.mats) == 1:
            return np.asarray(self.mats[0][idx])
        idx = np.asarray(idx)
        widths = [m.shape[1] for m in self.mats]
        out = np.empty((len(idx), sum(widths)), np.float32)
        for lo in range(0, len(idx), 1_000_000):
            sl = idx[lo:lo + 1_000_000]
            c0 = 0
            for m, w in zip(self.mats, widths):
                out[lo:lo + len(sl), c0:c0 + w] = m[sl]
                c0 += w
        return out

    def __len__(self):
        return len(self.mats[0])


def fit_xgb(X, y, idx_tr, idx_es, a, log, rounds=None):
    import xgboost as xgb
    params = dict(XGB_PARAMS, device=a.device, seed=a.seed)
    dtr = xgb.QuantileDMatrix(X[idx_tr], label=y[idx_tr], max_bin=params["max_bin"])
    evals, kw = [], {}
    if idx_es is not None:
        des = xgb.QuantileDMatrix(X[idx_es], label=y[idx_es], ref=dtr)
        evals, kw = [(des, "early_stop")], {"early_stopping_rounds": 50}
    t0 = time.time()
    bst = xgb.train(params, dtr, num_boost_round=rounds or a.max_rounds, evals=evals, verbose_eval=False, **kw)
    best = bst.best_iteration + 1 if idx_es is not None else rounds
    log(f"      trained on {len(idx_tr):,} pairs ({y[idx_tr].mean():.3%} positive), {best} rounds, "
        f"{time.time() - t0:,.0f}s")
    return bst, best


def predict(bst, X, idx, chunk=2_000_000):
    import xgboost as xgb
    out = np.empty(len(idx), np.float32)
    for lo in range(0, len(idx), chunk):
        sl = idx[lo:lo + chunk]
        out[lo:lo + len(sl)] = bst.predict(xgb.DMatrix(X[sl]), iteration_range=(0, bst.best_iteration + 1)
                                           if bst.attr("best_iteration") is not None else (0, 0))
    return out


def train_models(a, log, stage=1):
    """stage 1: pair features only; stage 2: pair features + collective features (built from the
    stage-1 out-of-fold probabilities).  Same out-of-fold protocol for both."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(a.seed)
    mats = [np.load(os.path.join(a.feat, "train_X.npy"), mmap_mode="r")]
    names = list(FEATURES)
    if stage >= 2:
        mats.append(np.load(os.path.join(a.feat, f"train_X2{a.x2_tag}.npy"), mmap_mode="r"))
        names += FEATURES2
    if stage == 3:
        mats.append(np.load(os.path.join(a.feat, "train_X3.npy"), mmap_mode="r"))
        names += FEATURES3
    X = Rows(*mats)
    y = np.load(os.path.join(a.feat, "train_y.npy"))
    fold = np.load(os.path.join(a.feat, "train_fold.npy"))
    groups = entity_groups("train", a)
    model_dir = os.path.join({1: a.model, 2: a.model2, 3: a.model3}[stage], "")
    os.makedirs(model_dir, exist_ok=True)
    oof = np.full(len(y), np.nan, np.float32)
    best_rounds = []
    for k in range(5):
        held = np.nonzero(fold == k)[0]
        fold_model = os.path.join(model_dir, f"xgb_fold{k}.json")
        fold_oof = os.path.join(model_dir, f"oof_fold{k}.npy")
        if os.path.exists(fold_model) and os.path.exists(fold_oof):     # resumable: fold already done
            import xgboost as xgb
            oof[held] = np.load(fold_oof)
            b = xgb.Booster(model_file=fold_model)
            best_rounds.append(int(b.attr("best_iteration")) + 1)
            log(f"  fold {k}: loaded from disk ({best_rounds[-1]} rounds)")
            continue
        log(f"  fold {k}: training on folds != {k}")
        rest = fold != k
        pick = sample_rows(groups, rest, a.train_frac + a.es_frac, rng)
        # split the sampled entities into training and early-stopping parts (by entity)
        ents = np.unique(groups[pick])
        es_ents = np.zeros(groups.max() + 1, bool)
        es_ents[rng.choice(ents, size=int(len(ents) * a.es_frac / (a.train_frac + a.es_frac)), replace=False)] = True
        idx_es, idx_tr = pick[es_ents[groups[pick]]], pick[~es_ents[groups[pick]]]
        bst, best = fit_xgb(X, y, idx_tr, idx_es, a, log)
        best_rounds.append(best)
        bst.save_model(fold_model)
        oof[held] = predict(bst, X, held)
        np.save(fold_oof, oof[held])
        log(f"      fold {k} out-of-fold: pair AUC {roc_auc_score(y[held], oof[held]):.5f}, "
            f"AP {average_precision_score(y[held], oof[held]):.5f}")
    np.save(os.path.join(model_dir, "train_oof.npy"), oof)
    log(f"  out-of-fold all train: AUC {roc_auc_score(y, oof):.5f}, AP {average_precision_score(y, oof):.5f}")
    if a.test_ensemble:
        # test = average of the 5 fold models (bagging; each saw a different entity sample)
        import xgboost as xgb
        del X, mats
        tm = [np.load(os.path.join(a.feat, "test_X.npy"), mmap_mode="r")]
        if stage >= 2:
            tm.append(np.load(os.path.join(a.feat, f"test_X2{a.x2_tag}.npy"), mmap_mode="r"))
        if stage == 3:
            tm.append(np.load(os.path.join(a.feat, "test_X3.npy"), mmap_mode="r"))
        Xt = Rows(*tm)
        pred = np.zeros(len(Xt), np.float64)
        for k in range(5):
            b = xgb.Booster(model_file=os.path.join(model_dir, f"xgb_fold{k}.json"))
            pred += predict(b, Xt, np.arange(len(Xt)))
            log(f"  test: fold-{k} model predicted")
        imp = b.get_score(importance_type="gain")
        with open(os.path.join(model_dir, "feature_importance.txt"), "w") as fh:
            for fname, g in sorted(((names[int(k[1:])], v) for k, v in imp.items()), key=lambda t: -t[1]):
                fh.write(f"{fname}\t{g:.2f}\n")
        np.save(os.path.join(model_dir, "test_pred.npy"), (pred / 5).astype(np.float32))
        log(f"  test predictions (mean of 5 fold models) -> {model_dir}test_pred.npy")
        return
    # final model for test: sample of all train entities, rounds = mean best rounds * 1.1
    rounds = int(np.mean(best_rounds) * 1.1)
    log(f"  final model (all folds, {rounds} rounds)")
    pick = sample_rows(groups, np.ones(len(y), bool), a.train_frac, rng)
    bst, _ = fit_xgb(X, y, pick, None, a, log, rounds=rounds)
    bst.save_model(os.path.join(model_dir, "xgb_final.json"))
    imp = bst.get_score(importance_type="gain")
    with open(os.path.join(model_dir, "feature_importance.txt"), "w") as fh:
        for fname, g in sorted(((names[int(k[1:])], v) for k, v in imp.items()), key=lambda t: -t[1]):
            fh.write(f"{fname}\t{g:.2f}\n")
    del X, mats
    tm = [np.load(os.path.join(a.feat, "test_X.npy"), mmap_mode="r")]
    if stage >= 2:
        tm.append(np.load(os.path.join(a.feat, f"test_X2{a.x2_tag}.npy"), mmap_mode="r"))
    if stage == 3:
        tm.append(np.load(os.path.join(a.feat, "test_X3.npy"), mmap_mode="r"))
    Xt = Rows(*tm)
    np.save(os.path.join(model_dir, "test_pred.npy"), predict(bst, Xt, np.arange(len(Xt))))
    log(f"  test predictions -> {model_dir}test_pred.npy")


# ========================================================================================
# stage 2: collective features from the stage-1 probabilities
# ========================================================================================
FEATURES2 = [
    # entity context (stage-1 probabilities of the entity's candidates)
    "p1", "e_rank_p1", "e_sum_p1", "e_n_conf", "e_max_other_p1",
    # the same Source-2/3 record seen from its competing entities
    "c_max_other_p1", "c_rank_p1", "c_n_conf",
    "gap_n_tset", "gap_n_ratio", "gap_n_jw", "gap_a_tset", "gap_u_tset", "gap_house",
    # agreement with the entity's most confident OTHER match ("sibling")
    "sib_p1", "sib_n_tset", "sib_n_ratio", "sib_p_ratio", "sib_c_eq", "sib_a_tset", "sib_u_tset", "sib_house_eq",
]


def _top2(key, value):
    """per group: row of the best value and best/second-best value (for 'max over the OTHER rows')"""
    o = np.lexsort((-value, key))
    k = key[o]
    starts = np.r_[0, np.nonzero(np.diff(k))[0] + 1]
    sizes = np.diff(np.r_[starts, len(k)])
    best_row = o[starts]
    best = value[o[starts]]
    second = np.where(sizes > 1, value[o[np.minimum(starts + 1, len(k) - 1)]], -1.0)
    second_row = np.where(sizes > 1, o[np.minimum(starts + 1, len(k) - 1)], -1)
    grp = np.empty(len(key), np.int64)
    grp[o] = np.repeat(np.arange(len(starts)), sizes)
    rank = np.empty(len(key), np.float32)
    rank[o] = np.arange(len(k)) - np.repeat(starts, sizes)
    return grp, best_row, best, second, second_row, rank


def _max_other(key, value):
    grp, best_row, best, second, _, rank = _top2(key, value)
    is_best = best_row[grp] == np.arange(len(key))
    return np.where(is_best, second[grp], best[grp]).astype(np.float32), rank


def build_stage2(split, a, log):
    t0 = time.time()
    X = np.load(os.path.join(a.feat, f"{split}_X.npy"), mmap_mode="r")
    col = {c: j for j, c in enumerate(FEATURES)}
    # probabilities the collective features are built from: stage 1 (default) or a later model
    # (--p1_model, for a second round of collective refinement)
    p_dir = a.p1_model or a.model
    p1 = np.load(os.path.join(p_dir, "train_oof.npy" if split == "train" else "test_pred.npy")).astype(np.float32)
    log(f"  [{split}] collective features from probabilities in {p_dir}")
    cand = read_arrow(os.path.join(a.blocks, f"{split}_candidates.tsv"),
                      {"source1_entity_id": pa.string(), "candidate_entity_id": pa.string()})
    _, pool = load_clean_tables(split, a.clean)
    s1col = cand["source1_entity_id"]
    s1col = s1col.combine_chunks() if isinstance(s1col, pa.ChunkedArray) else s1col
    g = pc.dictionary_encode(s1col).indices.to_numpy(zero_copy_only=False).astype(np.int64)
    q = pc.index_in(cand["candidate_entity_id"], value_set=pool["entity_id"]).to_numpy(zero_copy_only=False).astype(np.int64)
    n = len(p1)
    assert len(g) == n == len(X), "candidates / features / probabilities are not aligned"
    # needed stage-1 feature columns, read in row chunks (one pass over the memory-mapped file)
    need = ["n_tset", "n_ratio", "n_jw", "a_tset", "u_tset", "house_eq"]
    V = {c: np.empty(n, np.float32) for c in need}
    for lo in range(0, n, 2_000_000):
        blk = np.asarray(X[lo:lo + 2_000_000])
        for c in need:
            V[c][lo:lo + len(blk)] = blk[:, col[c]]
    F = {}
    F["p1"] = p1
    # --- entity context
    F["e_max_other_p1"], F["e_rank_p1"] = _max_other(g, p1)
    F["e_sum_p1"] = np.bincount(g, weights=p1)[g].astype(np.float32)
    F["e_n_conf"] = np.bincount(g, weights=p1 > 0.5)[g].astype(np.float32)
    # --- competing entities for the same record
    F["c_max_other_p1"], F["c_rank_p1"] = _max_other(q, p1)
    F["c_n_conf"] = np.bincount(q, weights=p1 > 0.5, minlength=q.max() + 1)[q].astype(np.float32)
    for f2, f1 in [("gap_n_tset", "n_tset"), ("gap_n_ratio", "n_ratio"), ("gap_n_jw", "n_jw"),
                   ("gap_a_tset", "a_tset"), ("gap_u_tset", "u_tset"), ("gap_house", "house_eq")]:
        v = V[f1]
        other, _ = _max_other(q, v)
        F[f2] = v - other
    log(f"  [{split}] context + competitor features ({time.time() - t0:,.0f}s)")
    # --- sibling: the entity's most confident other candidate (only if its p1 > 0.5)
    grp, best_row, best, second, second_row, _ = _top2(g, p1)
    rows = np.arange(n)
    sib = np.where(best_row[grp] == rows, second_row[grp], best_row[grp])
    sib_p = np.where(sib >= 0, p1[np.maximum(sib, 0)], -1.0).astype(np.float32)
    has = (sib >= 0) & (sib_p > 0.5)
    F["sib_p1"] = sib_p
    for c in ["sib_n_tset", "sib_n_ratio", "sib_p_ratio", "sib_c_eq", "sib_a_tset", "sib_u_tset", "sib_house_eq"]:
        F[c] = np.full(n, -1.0, np.float32)
    idx = np.nonzero(has)[0]
    for lo in range(0, len(idx), a.chunk):
        r = idx[lo:lo + a.chunk]
        qa, qb = pa.array(q[r]), pa.array(q[sib[r]])
        A = {c: pc.take(pool[c], qa) for c in ["name_core", "name_phon", "name_concat", "addr_words", "addr_nums", "house_no"]}
        B = {c: pc.take(pool[c], qb) for c in A}
        L = {c: v.to_pylist() for c, v in A.items()}
        R = {c: v.to_pylist() for c, v in B.items()}
        F["sib_n_tset"][r] = _cp(L["name_core"], R["name_core"], fuzz.token_set_ratio)
        F["sib_n_ratio"][r] = _cp(L["name_core"], R["name_core"], fuzz.ratio)
        F["sib_p_ratio"][r] = _cp(L["name_phon"], R["name_phon"], fuzz.ratio)
        F["sib_c_eq"][r] = _tri_state(A["name_concat"], B["name_concat"])
        a_ts = _cp(L["addr_words"], R["addr_words"], fuzz.token_set_ratio)
        u_ts = _cp(L["addr_nums"], R["addr_nums"], fuzz.token_set_ratio)
        a_miss = pc.or_(pc.equal(A["addr_words"], ""), pc.equal(B["addr_words"], "")).to_numpy(zero_copy_only=False)
        u_miss = pc.or_(pc.equal(A["addr_nums"], ""), pc.equal(B["addr_nums"], "")).to_numpy(zero_copy_only=False)
        a_ts[a_miss] = -1.0
        u_ts[u_miss] = -1.0
        F["sib_a_tset"][r], F["sib_u_tset"][r] = a_ts, u_ts
        F["sib_house_eq"][r] = _tri_state(A["house_no"], B["house_no"])
        log(f"    siblings {min(lo + a.chunk, len(idx)):,}/{len(idx):,} ({time.time() - t0:,.0f}s)")
    X2 = np.lib.format.open_memmap(os.path.join(a.feat, f"{split}_X2{a.x2_tag}.npy"), mode="w+", dtype=np.float32,
                                   shape=(n, len(FEATURES2)))
    for j, c in enumerate(FEATURES2):
        X2[:, j] = F[c]
    X2.flush()
    log(f"  [{split}] stage-2 features {X2.shape} ({time.time() - t0:,.0f}s)")


# ========================================================================================
# extra pair features (house-number arithmetic, rarity-weighted overlaps, source)
# ========================================================================================
FEATURES3 = ["src_c", "w_name_jac", "w_name_shared_max", "w_addr_jac", "w_street_jac", "nums_jac",
             "h_prefix", "h_suffix", "h_samelen_diff", "h_len_diff", "h_log_absdiff"]
_E = {}


def _init_extra(idf_name, idf_addr):
    _E["n"], _E["a"] = idf_name, idf_addr


def _wjac(x, y, idf):
    """rarity-weighted Jaccard of two token sets (-1 if one side is empty) and max weight shared"""
    if not x or not y:
        return -1.0, 0.0
    sx, sy = set(x.split()), set(y.split())
    inter = sx & sy
    wi = sum(idf.get(t, 12.0) for t in inter)
    wu = sum(idf.get(t, 12.0) for t in sx | sy)
    return wi / wu, max((idf.get(t, 12.0) for t in inter), default=0.0)


def _house(h1, h2):
    """prefix / suffix (truncation), same-length-but-different (neighbour), length diff, log |diff|"""
    if not h1 or not h2:
        return (-1.0,) * 5
    if h1 == h2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    pre = float(h1.startswith(h2) or h2.startswith(h1))
    suf = float(h1.endswith(h2) or h2.endswith(h1))
    same = float(len(h1) == len(h2))
    try:
        d = float(np.log1p(abs(int(h1) - int(h2))))
    except ValueError:
        d = -1.0
    return pre, suf, same, float(abs(len(h1) - len(h2))), d


def _extra_chunk(args):
    n1, n2, a1, a2, c1, c2, u1, u2, h1, h2 = args
    idf_n, idf_a = _E["n"], _E["a"]
    out = np.empty((len(n1), len(FEATURES3) - 1), np.float32)
    for i in range(len(n1)):
        wn, wmax = _wjac(n1[i], n2[i], idf_n)
        wa, _ = _wjac(a1[i], a2[i], idf_a)
        # street words = address words that are not part of the city
        s1 = " ".join(t for t in a1[i].split() if t not in set(c1[i].split()))
        s2 = " ".join(t for t in a2[i].split() if t not in set(c2[i].split()))
        ws, _ = _wjac(s1, s2, idf_a)
        if u1[i] and u2[i]:
            x, y = set(u1[i].split()), set(u2[i].split())
            nj = len(x & y) / len(x | y)
        else:
            nj = -1.0
        out[i] = (wn, wmax, wa, ws, nj) + _house(h1[i], h2[i])
    return out


def _idf(col, n):
    lists = pc.split_pattern(col, " ")
    flat = pc.list_flatten(lists)
    rec = pc.list_parent_indices(lists)
    keep = pc.not_equal(flat, "")
    t = pa.table({"t": pc.filter(flat, keep), "r": pc.filter(rec, keep)})
    df = t.group_by("t").aggregate([("r", "count_distinct")])
    return dict(zip(df["t"].to_pylist(), np.log(n / df["r_count_distinct"].to_numpy()).astype(float).tolist()))


def build_extra(split, a, log):
    from multiprocessing import Pool
    t0 = time.time()
    s1, pool = load_clean_tables(split, a.clean)
    cand = load_candidates(split, a.blocks)
    i1 = pc.index_in(cand["source1_entity_id"], value_set=s1["entity_id"]).to_numpy(zero_copy_only=False).astype(np.int64)
    i2 = pc.index_in(cand["candidate_entity_id"], value_set=pool["entity_id"]).to_numpy(zero_copy_only=False).astype(np.int64)
    idf_n, idf_a = _idf(pool["name_core"], len(pool)), _idf(pool["addr_words"], len(pool))
    n = len(cand)
    log(f"  [{split}] extra features for {n:,} pairs (vocab name {len(idf_n):,}, address {len(idf_a):,})")
    X3 = np.lib.format.open_memmap(os.path.join(a.feat, f"{split}_X3.npy"), mode="w+", dtype=np.float32,
                                   shape=(n, len(FEATURES3)))
    src = pc.starts_with(cand["candidate_entity_id"], "S3").to_numpy(zero_copy_only=False).astype(np.float32) + 2
    X3[:, 0] = src
    cols = ["name_core", "addr_words", "city", "addr_nums", "house_no"]
    step = 100_000
    workers = max(1, (os.cpu_count() or 2) - 1)
    with Pool(workers, initializer=_init_extra, initargs=(idf_n, idf_a)) as pp:
        for lo in range(0, n, a.chunk):
            hi = min(lo + a.chunk, n)
            A = {c: pc.take(s1[c], pa.array(i1[lo:hi])).to_pylist() for c in cols}
            B = {c: pc.take(pool[c], pa.array(i2[lo:hi])).to_pylist() for c in cols}
            jobs = []
            for s in range(0, hi - lo, step):
                e = s + step
                jobs.append((A["name_core"][s:e], B["name_core"][s:e], A["addr_words"][s:e], B["addr_words"][s:e],
                             A["city"][s:e], B["city"][s:e], A["addr_nums"][s:e], B["addr_nums"][s:e],
                             A["house_no"][s:e], B["house_no"][s:e]))
            X3[lo:hi, 1:] = np.vstack(pp.map(_extra_chunk, jobs))
            log(f"    {hi:,}/{n:,} ({time.time() - t0:,.0f}s)")
    X3.flush()
    log(f"  [{split}] extra features {X3.shape} ({time.time() - t0:,.0f}s)")


def main():
    ap = argparse.ArgumentParser(description="Matching model (Step 4)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--phase", required=True, choices=["features", "train", "stage2", "train2", "extra", "train3"])
    ap.add_argument("--clean", default=CLEAN)
    ap.add_argument("--blocks", default=BLOCK)
    ap.add_argument("--feat", default=FEAT)
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--model", default=os.path.join(ROOT, "artifacts", "model"))
    ap.add_argument("--model2", default=os.path.join(ROOT, "artifacts", "model_stage2"))
    ap.add_argument("--model3", default=os.path.join(ROOT, "artifacts", "model_stage3"))
    ap.add_argument("--device", default="cuda", help="cuda (GPU) or cpu")
    ap.add_argument("--train_frac", type=float, default=0.25, help="share of entities used to fit each model")
    ap.add_argument("--es_frac", type=float, default=0.03, help="share of entities held out for early stopping")
    ap.add_argument("--max_rounds", type=int, default=2000)
    ap.add_argument("--p1_model", default=None,
                    help="stage2 phase: model folder whose train_oof/test_pred feed the collective features "
                         "(default: the stage-1 --model)")
    ap.add_argument("--x2_tag", default="", help="suffix of the collective feature files, e.g. _r2")
    ap.add_argument("--test_ensemble", action="store_true",
                    help="predict test with the mean of the 5 fold models instead of one extra final model")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.feat, exist_ok=True)
    logf = open(os.path.join(a.feat, "matching_log.txt"), "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    if a.phase == "features":
        for split in a.splits.split(","):
            build_features(split, a, log)
    elif a.phase == "train":
        train_models(a, log, stage=1)
    elif a.phase == "stage2":
        for split in a.splits.split(","):
            build_stage2(split, a, log)
    elif a.phase == "extra":
        for split in a.splits.split(","):
            build_extra(split, a, log)
    elif a.phase == "train2":
        train_models(a, log, stage=2)
    else:
        train_models(a, log, stage=3)


if __name__ == "__main__":
    main()
