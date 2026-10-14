"""Where are the F0.5 points lost?  (out-of-fold train predictions vs train_ground_truth.tsv)

Usage: python src/error_analysis.py --data ../../dataset --model artifacts/model_stage3b [--rule expected --param 1.0]
Loss is split into false merges, threshold misses (true pair in candidates, not predicted) and blocking
misses (true pair never a candidate), then broken down by entity / record segment, with raw examples.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decide import BLOCK, FEAT, expected_f05_choice, load_pairs, one_to_one, read_arrow  # noqa: E402
from matching import FEATURES, FEATURES2, FEATURES3  # noqa: E402

TYPES = {0: "normal", 1: "domain", 2: "handle", 3: "brand", 4: "alias", 5: "acronym"}


def f05(tp, npred, ntrue):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ntrue == 0, (npred == 0).astype(float), np.where(tp > 0, 1.25 * tp / (0.25 * ntrue + npred), 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rule", default="expected", choices=["expected", "threshold"])
    ap.add_argument("--param", type=float, default=1.0)
    ap.add_argument("--examples", type=int, default=10)
    a = ap.parse_args()
    c, g, s1_dict, q, pool_dict = load_pairs("train", BLOCK)
    p = np.load(os.path.join(a.model, "train_oof.npy"))
    y = np.load(os.path.join(FEAT, "train_y.npy")).astype(bool)
    p11 = one_to_one(q, p)
    pred = expected_f05_choice(g, p11, a.param) if a.rule == "expected" else p11 >= a.param
    gt = read_arrow(os.path.join(a.data, "train", "train_ground_truth.tsv"),
                    {"source1_entity_id": pa.string(), "matched_entity_ids": pa.string()})
    n_gt = pc.list_value_length(pc.split_pattern(gt["matched_entity_ids"], ",")).to_numpy(zero_copy_only=False)
    n_gt = np.where(pc.equal(gt["matched_entity_ids"], "").to_numpy(zero_copy_only=False), 0, n_gt)
    pos = pc.index_in(s1_dict, value_set=gt["source1_entity_id"]).to_numpy(zero_copy_only=False).astype(np.int64)
    ne = len(s1_dict)
    ntrue = n_gt[pos].astype(float)
    tp = np.bincount(g, weights=pred & y, minlength=ne)
    fp = np.bincount(g, weights=pred & ~y, minlength=ne)
    tp_cand = np.bincount(g, weights=y, minlength=ne)
    F = f05(tp, tp + fp, ntrue)
    F_noFP = f05(tp, tp, ntrue)
    F_allcand = f05(tp_cand, tp_cand, ntrue)
    loss = 1 - F.mean()
    l_fp, l_thr, l_blk = (F_noFP - F).mean(), (F_allcand - F_noFP).mean(), (1 - F_allcand).mean()
    print(f"macro F0.5 {F.mean():.5f} -> loss {loss:.5f}: false merges {l_fp:.5f} ({l_fp / loss:.0%}) | "
          f"threshold misses {l_thr:.5f} ({l_thr / loss:.0%}) | blocking misses {l_blk:.5f} ({l_blk / loss:.0%})")

    # ---------------- entity segments ----------------
    starts = np.r_[0, np.nonzero(np.diff(g))[0] + 1]
    ent_country = np.empty(ne, object)
    ent_country[g[starts]] = pc.take(c["country"], pa.array(starts)).to_numpy(zero_copy_only=False)
    X = np.load(os.path.join(FEAT, "train_X.npy"), mmap_mode="r")
    col = {f: i for i, f in enumerate(FEATURES)}
    # entity-level S1 properties from the first candidate row of each entity
    first = np.asarray(X[starts])
    ent_addr_tok = np.empty(ne, np.float32); ent_addr_tok[g[starts]] = first[:, col["a_tok_s1"]]
    ent_name_tok = np.empty(ne, np.float32); ent_name_tok[g[starts]] = first[:, col["n_tok_s1"]]
    seg = pd.DataFrame({"country": ent_country, "n_true": np.minimum(ntrue, 6).astype(int),
                        "s1_name_words": np.minimum(ent_name_tok, 5).astype(int),
                        "s1_addr_words": pd.cut(ent_addr_tok, [-1, 2, 4, 7, 100], labels=["<=2", "3-4", "5-7", "8+"]),
                        "l_fp": F_noFP - F, "l_thr": F_allcand - F_noFP, "l_blk": 1 - F_allcand})
    tot_loss = loss * ne
    print("\nENTITY SEGMENTS: mean loss per entity (fp / thr / blk) and share of total loss")
    for key in ["country", "n_true", "s1_name_words", "s1_addr_words"]:
        for k, h in seg.groupby(key, observed=True):
            s = (h.l_fp.sum() + h.l_thr.sum() + h.l_blk.sum()) / tot_loss
            print(f"  {key}={k!s:5s} n={len(h):>9,} | fp {h.l_fp.mean():.4f} thr {h.l_thr.mean():.4f} "
                  f"blk {h.l_blk.mean():.4f} | share {s:.1%}")

    # ---------------- pair-level error segments ----------------
    X2 = np.load(os.path.join(FEAT, "train_X2.npy"), mmap_mode="r")
    X3 = np.load(os.path.join(FEAT, "train_X3.npy"), mmap_mode="r")
    c2 = {f: i for i, f in enumerate(FEATURES2)}
    c3 = {f: i for i, f in enumerate(FEATURES3)}
    fp_idx = np.nonzero(pred & ~y)[0]
    fn_idx = np.nonzero(~pred & y)[0]
    tp_idx = np.nonzero(pred & y)[0]
    rng = np.random.default_rng(0)
    print(f"\npairs: true positives {len(tp_idx):,} | false merges {len(fp_idx):,} | threshold misses {len(fn_idx):,}")

    def describe(idx, name):
        smp = np.sort(rng.choice(idx, size=min(150000, len(idx)), replace=False))
        A, B2, B3 = np.asarray(X[smp]), np.asarray(X2[smp]), np.asarray(X3[smp])
        d = pd.DataFrame({
            "type": pd.Series(A[:, col["c_type"]]).map(TYPES), "addr_empty": A[:, col["addr_empty_c"]],
            "native": A[:, col["c_native"]], "src": B3[:, c3["src_c"]], "n_tset": A[:, col["n_tset"]],
            "a_tset": A[:, col["a_tset"]], "house": A[:, col["house_eq"]], "h_same": B3[:, c3["h_samelen_diff"]],
            "h_trunc": np.maximum(B3[:, c3["h_prefix"]], B3[:, c3["h_suffix"]]), "state": A[:, col["state_eq"]],
            "city": A[:, col["city_eq"]], "unit": A[:, col["unit_eq"]], "legal": A[:, col["legal_eq"]],
            "c_n_s1": A[:, col["c_n_s1"]], "sib": B2[:, c2["sib_p1"]], "sib_house": B2[:, c2["sib_house_eq"]],
            "e_n_conf": B2[:, c2["e_n_conf"]], "wn": B3[:, c3["w_name_jac"]], "p": p[smp]})
        d["country"] = pc.take(c["country"], pa.array(smp)).to_numpy(zero_copy_only=False)
        print(f"\n== {name} (sample {len(d):,})")
        print(f"  country {d.country.value_counts(normalize=True).round(3).to_dict()} | "
              f"source S3 share {(d.src == 3).mean():.3f}")
        print(f"  record type {d.type.value_counts(normalize=True).round(3).to_dict()}")
        print(f"  candidate address empty {d.addr_empty.mean():.3f} | native script {d.native.mean():.3f}")
        print(f"  house: equal {(d.house == 1).mean():.3f} differ {(d.house == 0).mean():.3f} missing "
              f"{(d.house == -1).mean():.3f} | same-length-differ {(d.h_same == 1).mean():.3f} truncation "
              f"{(d.h_trunc == 1).mean():.3f}")
        print(f"  unit equal/differ/missing {(d.unit == 1).mean():.3f}/{(d.unit == 0).mean():.3f}/{(d.unit == -1).mean():.3f}"
              f" | city equal {(d.city == 1).mean():.3f} | state equal {(d.state == 1).mean():.3f} | legal equal {(d.legal == 1).mean():.3f}")
        print(f"  name token-set: =100 {(d.n_tset == 100).mean():.3f}, <80 {(d.n_tset < 80).mean():.3f} | "
              f"weighted name jaccard median {d.wn.median():.2f} | addr token-set median {d.a_tset.median():.0f}")
        print(f"  record claimed by >1 entity {(d.c_n_s1 > 1).mean():.3f} | entity has confident sibling "
              f"{(d.sib > 0.5).mean():.3f} (sibling house equal {(d.sib_house == 1).mean():.3f}) | "
              f"entity confident matches median {d.e_n_conf.median():.0f}")
        print(f"  probability quantiles 10/50/90%: {np.quantile(d.p, [.1, .5, .9]).round(3).tolist()}")
        return d

    describe(tp_idx, "TRUE POSITIVES (reference)")
    describe(fp_idx, "FALSE MERGES")
    describe(fn_idx, "THRESHOLD MISSES")

    # ---------------- raw examples ----------------
    raw = {}
    for k in "123":
        r = read_arrow(os.path.join(a.data, "train", f"train_source{k}.tsv"),
                       {"entity_id": pa.string(), "business_name": pa.string(), "business_address": pa.string()})
        raw.update(zip(r["entity_id"].to_pylist(), zip(r["business_name"].to_pylist(), r["business_address"].to_pylist())))
    s1c, cc = c["source1_entity_id"], c["candidate_entity_id"]
    for name, idx in [("FALSE MERGE", fp_idx), ("THRESHOLD MISS", fn_idx)]:
        print(f"\n######## examples: {name}")
        for i in rng.choice(idx, size=a.examples, replace=False):
            i = int(i)
            e1, e2 = s1c[i].as_py(), cc[i].as_py()
            # the entity's confident predicted matches, for context
            rows = np.nonzero(g == g[i])[0]
            conf = [cc[int(r)].as_py() for r in rows if pred[r] and r != i][:2]
            print(f"  p={p[i]:.3f} [{c['country'][i].as_py()}]\n    S1  {raw[e1]}\n    {e2[:2]}  {raw[e2]}")
            for e3 in conf:
                print(f"      entity's other predicted match: {raw[e3]} {'(TRUE)' if y[rows[[cc[int(r)].as_py() for r in rows].index(e3)]] else '(false)'}")


if __name__ == "__main__":
    main()
