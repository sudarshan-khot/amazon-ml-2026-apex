"""Assign every TRAIN Source-1 entity to one of K folds (grouped + stratified).

* Unit of split = Source-1 entity; all of its true Source-2/3 matches get the same fold.
* Stratified on (country, number-of-matches bucket) so every fold has the same singleton rate
  and difficulty mix.
* Unmatched Source-2/3 records (distractors) get fold -1: they stay in the search pool of every
  fold, exactly like distractors in the test set.
Output: artifacts/folds.tsv  (entity_id, source, fold)
Usage:  python src/make_folds.py --data ../../dataset --k 5 --seed 42
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))


def rd(p, cols):
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, usecols=cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "artifacts", "folds.tsv"))
    a = ap.parse_args()
    d = os.path.join(a.data, "train")

    gt = rd(os.path.join(d, "train_ground_truth.tsv"), ["source1_entity_id", "matched_entity_ids"])
    s1 = rd(os.path.join(d, "train_source1.tsv"), ["entity_id", "country"])
    gt["matches"] = gt.matched_entity_ids.map(lambda s: [x for x in s.split(",") if x])
    gt["n"] = gt.matches.map(len)
    df = s1.merge(gt, left_on="entity_id", right_on="source1_entity_id", how="left", validate="1:1")
    assert df.n.notna().all(), "every Source-1 entity must be in the ground truth"
    df["bucket"] = df.n.clip(upper=5).astype(int).astype(str)
    strat = df.country + "_" + df.bucket

    skf = StratifiedKFold(n_splits=a.k, shuffle=True, random_state=a.seed)
    fold = np.full(len(df), -1)
    for f, (_, idx) in enumerate(skf.split(df, strat)):
        fold[idx] = f
    df["fold"] = fold

    rows = [pd.DataFrame({"entity_id": df.entity_id, "source": "S1", "fold": df.fold})]
    m = df[["fold", "matches"]].explode("matches").dropna()
    rows.append(pd.DataFrame({"entity_id": m.matches.values, "source": m.matches.str[:2].values,
                              "fold": m.fold.values}))
    for k in "23":
        ids = rd(os.path.join(d, f"train_source{k}.tsv"), ["entity_id"]).entity_id
        matched = set(m.matches)
        orphan = ids[~ids.isin(matched)]
        rows.append(pd.DataFrame({"entity_id": orphan.values, "source": f"S{k}", "fold": -1}))
    out = pd.concat(rows, ignore_index=True)
    out["fold"] = out.fold.astype("int8")

    # ---------------- integrity checks ----------------
    assert out.entity_id.is_unique, "a record was assigned to two folds"
    n_s23 = sum(len(rd(os.path.join(d, f"train_source{k}.tsv"), ["entity_id"])) for k in "23")
    assert (out.source != "S1").sum() == n_s23, "every Source-2/3 record must appear exactly once"
    assert (out.source == "S1").sum() == len(s1)
    print(f"records: S1 {len(s1):,} | S2/S3 {n_s23:,} (distractors, fold -1: {(out.fold == -1).sum():,})")
    summ = df.groupby("fold").agg(entities=("entity_id", "size"), us=("country", lambda x: (x == "US").mean()),
                                  singletons=("n", lambda x: (x == 0).mean()), mean_matches=("n", "mean"))
    print(summ.round(4).to_string())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, sep="\t", index=False, quoting=csv.QUOTE_NONE, lineterminator="\n")
    print("saved", a.out)


if __name__ == "__main__":
    main()
