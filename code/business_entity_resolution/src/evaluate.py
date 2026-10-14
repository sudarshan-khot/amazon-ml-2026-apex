"""Official-style scorer: macro F0.5 over ALL Source-1 entities (singletons included).

Usage: python src/evaluate.py --pred <matching_results.tsv> --truth <ground_truth.tsv> [--ids <subset file>]
A Source-1 entity with no true matches scores 1.0 for an empty prediction and 0.0 otherwise.
"""
import argparse
import csv

import pandas as pd

BETA2 = 0.25   # beta = 0.5


def parse_ids(s):
    return set(x for x in (s or "").split(",") if x)


def f05(pred, true):
    """F0.5 of one Source-1 entity."""
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def score(pred_map, true_map, ids=None):
    """pred_map / true_map: {s1_id: set(ids)}.  ids: entities to average over (default: all in truth)."""
    ids = list(true_map) if ids is None else list(ids)
    vals = [f05(pred_map.get(i, set()), true_map[i]) for i in ids]
    return sum(vals) / len(vals), vals


def read_map(path, col):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    return {k: parse_ids(v) for k, v in zip(df.source1_entity_id, df[col])}


def _self_test():
    # README worked example: predicted 3, 2 correct, 2 true -> 0.714
    v = f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(v - 0.7142857) < 1e-6, v
    assert f05(set(), set()) == 1.0                 # singleton, correctly empty
    assert f05({"S2-1"}, set()) == 0.0              # singleton, false merge
    assert f05(set(), {"S2-1"}) == 0.0              # missed everything
    assert f05({"S2-1"}, {"S2-1"}) == 1.0           # perfect
    assert f05({"S2-9"}, {"S2-1"}) == 0.0           # all wrong
    # precision weighted more than recall: P=1,R=.5 must beat P=.5,R=1
    assert f05({"a"}, {"a", "b"}) > f05({"a", "b"}, {"a"})
    m, _ = score({"x": set(), "y": {"a"}}, {"x": set(), "y": {"a", "b"}})
    assert abs(m - (1.0 + 0.8333333) / 2) < 1e-6, m
    print("scorer self-test passed (README example = %.3f)" % v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred")
    ap.add_argument("--truth")
    ap.add_argument("--ids", help="optional file with one source1_entity_id per line to restrict the average")
    a = ap.parse_args()
    _self_test()
    if a.pred and a.truth:
        truth = read_map(a.truth, "matched_entity_ids")
        pred = read_map(a.pred, "matched_entity_ids")
        ids = None
        if a.ids:
            ids = [l.strip() for l in open(a.ids) if l.strip()]
        m, _ = score(pred, truth, ids)
        print(f"macro F0.5 = {m:.5f} over {len(ids) if ids else len(truth):,} Source-1 entities")
