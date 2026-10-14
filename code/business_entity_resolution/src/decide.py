"""Decisions: pair probabilities -> final matches -- single entry point for Step 5.

Run from code/business_entity_resolution/ (after matching.py):

    python src/decide.py --data ../../dataset

1. On TRAIN (out-of-fold probabilities) several decision rules are scored with the exact challenge
   metric (macro F0.5 over every Source-1 entity, singletons included) and their parameters tuned:
     * threshold          predict every candidate with probability >= t
     * one-to-one         a Source-2/3 record may go only to the entity that scores it highest
                          (the ground truth never assigns a record to two entities)
     * expected-F0.5      per entity, predict the top-k candidates that maximise the expected F0.5
                          given the probabilities (k = 0 is valued at P(no true match))
2. The best rule is applied to the TEST probabilities -> output/matching_results.tsv, and the
   blocking candidate list is copied to output/candidate_pairs.tsv.
"""
import argparse
import csv
import os
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BLOCK = os.path.join(ROOT, "artifacts", "blocking")
MODEL = os.path.join(ROOT, "artifacts", "model")
FEAT = os.path.join(ROOT, "artifacts", "features")


def read_arrow(path, types):
    return pacsv.read_csv(
        path, parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(column_types=types, include_columns=list(types),
                                             strings_can_be_null=False)).combine_chunks()


def encode(col):
    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    enc = pc.dictionary_encode(col)
    return enc.indices.to_numpy(zero_copy_only=False).astype(np.int64), enc.dictionary


def load_pairs(split, blocks):
    c = read_arrow(os.path.join(blocks, f"{split}_candidates.tsv"),
                   {"source1_entity_id": pa.string(), "candidate_entity_id": pa.string(), "country": pa.string(),
                    "fold": pa.int8()})
    g, s1_dict = encode(c["source1_entity_id"])
    q, pool_dict = encode(c["candidate_entity_id"])
    return c, g, s1_dict, q, pool_dict


# ----------------------------------------------------------------------------------------
# decision rules (vectorised; rows of an entity are contiguous in the candidate files)
# ----------------------------------------------------------------------------------------
def one_to_one(q, p):
    """keep, for every pool record q, only its highest-probability entity (others -> 0)"""
    o = np.lexsort((-p, q))
    first = np.r_[True, np.diff(q[o]) != 0]
    keep = np.zeros(len(p), bool)
    keep[o[first]] = True
    return np.where(keep, p, 0.0).astype(np.float32)


def expected_f05_choice(g, p, gamma=1.0, extra_true=0.0):
    """per entity choose k (predict its k most probable candidates) maximising expected F0.5.
    E[F | k] ~ 1.25 * sum_topk p / (0.25 * (sum_all p + extra_true) + k),  E[F | 0] = prod(1 - p)."""
    p = np.clip(p.astype(np.float64) ** gamma, 0, 1)
    o = np.lexsort((-p, g))
    gs, ps = g[o], p[o]
    starts = np.r_[0, np.nonzero(np.diff(gs))[0] + 1]
    sizes = np.diff(np.r_[starts, len(gs)])
    rank = np.arange(len(gs)) - np.repeat(starts, sizes)
    csum = np.cumsum(ps)
    before = np.repeat(np.r_[0.0, csum[starts[1:] - 1]], sizes)
    cs_k = csum - before                                         # sum of top-(rank+1) probabilities
    total = np.repeat(np.add.reduceat(ps, starts), sizes) + extra_true
    e_k = 1.25 * cs_k / (0.25 * total + rank + 1)
    e0 = np.exp(np.add.reduceat(np.log1p(-np.minimum(ps, 1 - 1e-9)), starts))
    best_e = np.maximum.reduceat(e_k, starts)
    best_k = np.zeros(len(starts), np.int64)
    # k* = first rank reaching the group's max expected value (if it beats predicting nothing)
    hit = e_k >= np.repeat(best_e, sizes) - 1e-12
    first_hit = np.minimum.reduceat(np.where(hit, rank, 10 ** 9), starts)
    best_k = np.where(best_e > e0, first_hit + 1, 0)
    pred_sorted = rank < np.repeat(best_k, sizes)
    pred = np.empty(len(p), bool)
    pred[o] = pred_sorted
    return pred


def macro_f05(g, pred, y, n_true, n_ent):
    """exact challenge metric; n_true[e] = number of true matches of entity e (incl. ones blocking missed)"""
    npred = np.bincount(g, weights=pred, minlength=n_ent)
    tp = np.bincount(g, weights=pred & (y == 1), minlength=n_ent)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(n_true == 0, (npred == 0).astype(float),
                     np.where(tp > 0, 1.25 * tp / (0.25 * n_true + npred), 0.0))
    return f


def main():
    ap = argparse.ArgumentParser(description="Decision rules and final output (Step 5)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--blocks", default=BLOCK)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--feat", default=FEAT)
    ap.add_argument("--out", default=os.path.join(ROOT, "..", "..", "output"))
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    # ---------------- train: tune on out-of-fold probabilities ----------------
    c, g, s1_dict, q, _ = load_pairs("train", a.blocks)
    p = np.load(os.path.join(a.model, "train_oof.npy")).astype(np.float32)
    y = np.load(os.path.join(a.feat, "train_y.npy"))
    gt = read_arrow(os.path.join(a.data, "train", "train_ground_truth.tsv"),
                    {"source1_entity_id": pa.string(), "matched_entity_ids": pa.string()})
    gt_n = pc.list_value_length(pc.split_pattern(gt["matched_entity_ids"], ",")).to_numpy(zero_copy_only=False)
    gt_n = np.where(pc.equal(gt["matched_entity_ids"], "").to_numpy(zero_copy_only=False), 0, gt_n)
    pos = pc.index_in(s1_dict, value_set=gt["source1_entity_id"]).to_numpy(zero_copy_only=False).astype(np.int64)
    n_ent = len(s1_dict)
    n_true = gt_n[pos]
    assert n_ent == len(gt), "every train Source-1 entity must have candidates"
    country = pc.take(c["country"], pa.array(np.r_[0, np.nonzero(np.diff(g))[0] + 1])).to_numpy(zero_copy_only=False)
    ent_country = np.empty(n_ent, object)
    ent_country[g[np.r_[0, np.nonzero(np.diff(g))[0] + 1]]] = country
    ent_fold = np.empty(n_ent, np.int8)
    ent_fold[g] = c["fold"].to_numpy(zero_copy_only=False)
    log(f"train: {len(p):,} pairs, {n_ent:,} entities, {int(y.sum()):,} true pairs found by blocking "
        f"of {int(n_true.sum()):,}")

    p11 = one_to_one(q, p)
    probs = {"threshold": p, "one-to-one + threshold": p11,
             "expected-F0.5": p, "one-to-one + expected-F0.5": p11}

    def decide(name, param, pp):
        return expected_f05_choice(g, pp, param) if "expected" in name else pp >= param

    results = []                                   # (rule, parameter, macro F0.5); predictions not kept
    for name in ["threshold", "one-to-one + threshold"]:
        for t in np.round(np.arange(0.20, 0.96, 0.025), 3):
            results.append((name, float(t), macro_f05(g, probs[name] >= t, y, n_true, n_ent).mean()))
    for name in ["expected-F0.5", "one-to-one + expected-F0.5"]:
        for gamma in [0.8, 1.0, 1.2, 1.5, 2.0]:
            pred = expected_f05_choice(g, probs[name], gamma)
            results.append((name, gamma, macro_f05(g, pred, y, n_true, n_ent).mean()))
    log("rule comparison (macro F0.5 on out-of-fold train predictions):")
    for rule in dict.fromkeys(r[0] for r in results):
        r = max((r for r in results if r[0] == rule), key=lambda r: r[2])
        log(f"   {rule:30s} best parameter {r[1]:<6} F0.5 = {r[2]:.5f}")
    name, param, score = max(results, key=lambda r: r[2])
    pred = decide(name, param, probs[name])
    del probs
    f = macro_f05(g, pred, y, n_true, n_ent)
    tp = np.bincount(g, weights=pred & (y == 1), minlength=n_ent)
    npred = np.bincount(g, weights=pred, minlength=n_ent)
    log(f"BEST: {name} ({param}) -> macro F0.5 {score:.5f} | pair precision {tp.sum() / max(npred.sum(), 1):.4f}, "
        f"recall {tp.sum() / n_true.sum():.4f}")
    for ctry in ["US", "India"]:
        m = ent_country == ctry
        log(f"   {ctry:6s} F0.5 {f[m].mean():.5f}")
    for k in range(5):
        log(f"   fold {k} F0.5 {f[ent_fold == k].mean():.5f}")
    sing = n_true == 0
    log(f"   singletons: F0.5 {f[sing].mean():.5f} ({sing.sum():,}) | with matches: {f[~sing].mean():.5f}")

    # ---------------- test: apply the chosen rule ----------------
    ct, gt_, s1_dict_t, qt, pool_dict_t = load_pairs("test", a.blocks)
    pt = np.load(os.path.join(a.model, "test_pred.npy")).astype(np.float32)
    if name.startswith("one-to-one"):
        pt = one_to_one(qt, pt)
    pred_t = expected_f05_choice(gt_, pt, param) if "expected" in name else pt >= param
    s1_all = read_arrow(os.path.join(a.data, "test", "test_source1.tsv"), {"entity_id": pa.string()})["entity_id"]
    sel = np.nonzero(pred_t)[0]
    df = pd.DataFrame({"s": s1_dict_t.take(pa.array(gt_[sel])).to_numpy(zero_copy_only=False),
                       "m": pool_dict_t.take(pa.array(qt[sel])).to_numpy(zero_copy_only=False)})
    lists = df.groupby("s", sort=False).m.agg(",".join)
    out = pd.DataFrame({"source1_entity_id": s1_all.to_numpy(zero_copy_only=False)})
    out["matched_entity_ids"] = out.source1_entity_id.map(lists).fillna("")
    os.makedirs(a.out, exist_ok=True)
    out.to_csv(os.path.join(a.out, "matching_results.tsv"), sep="\t", index=False, quoting=csv.QUOTE_NONE,
               lineterminator="\n")
    shutil.copyfile(os.path.join(a.blocks, "candidate_pairs.tsv"), os.path.join(a.out, "candidate_pairs.tsv"))
    n_match = (out.matched_entity_ids != "").mean()
    log(f"test: {len(out):,} Source-1 rows, {len(sel):,} predicted matches, entities with >=1 match "
        f"{n_match:.3f} (train truth: {(n_true > 0).mean():.3f}) -> {a.out}")
    log(f"done in {time.time() - t0:,.0f}s")


if __name__ == "__main__":
    main()
