"""Integrity + sanity checks of the blocking outputs (train and test).

Usage: python src/check_blocking.py --data ../../dataset
"""
import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "artifacts", "blocking")


def read(path, types):
    return pacsv.read_csv(path, parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                          convert_options=pacsv.ConvertOptions(column_types=types, include_columns=list(types),
                                                               strings_can_be_null=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    s = pa.string()
    ok = True
    for split in ("train", "test"):
        c = read(os.path.join(OUT, f"{split}_candidates.tsv"),
                 {"source1_entity_id": s, "candidate_entity_id": s, "country": s, "rr_prob": pa.float64(),
                  "rr_rank": pa.int32(), "blk_wj": pa.float64()})
        s1 = read(os.path.join(a.data, split, f"{split}_source1.tsv"), {"entity_id": s, "country": s})
        pool = pa.concat_tables([read(os.path.join(a.data, split, f"{split}_source{k}.tsv"),
                                      {"entity_id": s, "country": s}) for k in "23"])
        msgs = []
        # pairs unique
        key = pc.binary_join_element_wise(c["source1_entity_id"], c["candidate_entity_id"], "|")
        if pc.count_distinct(key).as_py() != len(c):
            msgs.append("duplicate (S1, candidate) pairs")
        # ids exist and countries agree
        s1_country = dict(zip(s1["entity_id"].to_pylist(), s1["country"].to_pylist()))
        pool_country = dict(zip(pool["entity_id"].to_pylist(), pool["country"].to_pylist()))
        cs1, ccand, cc = (c[x].to_pylist() for x in ["source1_entity_id", "candidate_entity_id", "country"])
        miss1 = sum(1 for e in set(cs1) if e not in s1_country)
        miss2 = sum(1 for e in set(ccand) if e not in pool_country)
        cross = sum(1 for e1, e2 in zip(cs1, ccand) if s1_country.get(e1) != pool_country.get(e2))
        if miss1 or miss2:
            msgs.append(f"unknown ids: S1 {miss1}, candidates {miss2}")
        if cross:
            msgs.append(f"{cross} cross-country pairs")
        per = np.bincount(pc.dictionary_encode(c["source1_entity_id"]).combine_chunks().indices.to_numpy())
        if per.max() > 20:
            msgs.append(f"an S1 record has {per.max()} candidates (> 20)")
        # coverage and distribution per country
        print(f"\n== {split}: {len(c):,} pairs | S1 with >=1 candidate {len(per):,} / {len(s1):,} | "
              f"{'OK' if not msgs else 'FAIL: ' + '; '.join(msgs)}")
        top = c.filter(pc.equal(c["rr_rank"], 0))
        for ctry in sorted(set(s1["country"].to_pylist())):
            m = pc.equal(c["country"], ctry)
            tm = pc.equal(top["country"], ctry)
            n_s1 = pc.sum(pc.equal(s1["country"], ctry)).as_py()
            n_pairs = pc.sum(m).as_py()
            tp = top.filter(tm)["rr_prob"].to_numpy()
            wj = top.filter(tm)["blk_wj"].to_numpy()
            print(f"   {ctry:7s} S1 {n_s1:>9,} | pairs/S1 {n_pairs / n_s1:5.1f} | best-candidate re-rank prob: "
                  f"median {np.median(tp):.3f}, share >0.5 {np.mean(tp > 0.5):.3f} | best weighted-Jaccard median {np.median(wj):.3f}")
        ok &= not msgs
    print("\nALL BLOCKING CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()
