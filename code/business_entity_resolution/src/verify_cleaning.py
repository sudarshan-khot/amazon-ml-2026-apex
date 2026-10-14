"""Quantitative verification of the normaliser on TRAIN ground-truth pairs.

Compares raw text vs cleaned fields on
  * positives      : true (S1, S2/S3) pairs
  * hard negatives : same country, same raw last address part, sharing >= 1 raw name token,
                     but a different entity
  * random negatives (same country)
Usage: python src/verify_cleaning.py --data ../../dataset --n_s1 40000
"""
import argparse
import csv
import os
import pickle
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_cleaning import (NameNormalizer, NameResources, fold, normalize_address, phon,  # noqa: E402
                       transliterate_token)

HERE = os.path.dirname(os.path.abspath(__file__))


def rd(p, cols=None):
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, usecols=cols)


def raw_toks(s):
    return set(re.findall(r"[a-z0-9]+", fold(s)))


def jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_s1", type=int, default=40000)
    a = ap.parse_args()
    res = pickle.load(open(os.path.join(HERE, "..", "artifacts", "resources.pkl"), "rb"))
    NN = NameNormalizer(NameResources(res["vocab"], res["native_lex"], res["eng_phon"]))

    # ---------------- sample entities and their records ----------------
    d = os.path.join(a.data, "train")
    gt = rd(os.path.join(d, "train_ground_truth.tsv"))
    s1 = rd(os.path.join(d, "train_source1.tsv"))
    rng = random.Random(0)
    ids = set(rng.sample(list(s1.entity_id), a.n_s1))
    s1 = s1[s1.entity_id.isin(ids)].set_index("entity_id")
    link = {}
    for sid, ms in zip(gt.source1_entity_id, gt.matched_entity_ids):
        if sid in ids:
            for m in ms.split(","):
                if m:
                    link[m] = sid
    recs = [s1.reset_index().assign(s1=lambda x: x.entity_id)]
    for k in "23":
        x = rd(os.path.join(d, f"train_source{k}.tsv"))
        x = x[x.entity_id.isin(link)]
        x["s1"] = x.entity_id.map(link)
        recs.append(x)
    df = pd.concat(recs, ignore_index=True)
    print("records", len(df))

    t0 = time.time()
    nm = [NN.normalize(n, c) for n, c in zip(df.business_name, df.country)]
    ad = [normalize_address(x, c) for x, c in zip(df.business_address, df.country)]
    dt = time.time() - t0
    print(f"normalisation speed: {len(df) / dt:,.0f} records/s (single core)")
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(nm), pd.DataFrame(ad)], axis=1)
    df["src"] = df.entity_id.str[:2]
    df["raw_last"] = df.business_address.map(lambda s: fold(s.split(",")[-1].strip()) if s else "")
    pickle.dump(df, open(os.path.join(HERE, "..", "artifacts", "verify_sample.pkl"), "wb"))

    S1 = df[df.src == "S1"].set_index("s1")
    other = df[df.src != "S1"]

    # ---------------- pairs ----------------
    pos = [(s, i) for i, s in zip(other.index, other.s1)]
    by_key = defaultdict(list)
    for i, n, c, st in zip(other.index, other.business_name, other.country, other.state):
        for t in raw_toks(n):
            if len(t) >= 3:
                by_key[(c, st, t)].append(i)
    hard = []
    for s, r in S1.iterrows():
        cands = set()
        for t in raw_toks(r.business_name):
            if len(t) >= 3:
                for i in by_key.get((r.country, r.state, t), [])[:50]:
                    if other.at[i, "s1"] != s:
                        cands.add(i)
        cands = list(cands)
        rng.shuffle(cands)
        hard += [(s, i) for i in cands[:3]]
    by_c = defaultdict(list)
    for i, c in zip(other.index, other.country):
        by_c[c].append(i)
    rand = []
    for s, r in S1.iterrows():
        for _ in range(3):
            i = rng.choice(by_c[r.country])
            if other.at[i, "s1"] != s:
                rand.append((s, i))
    print(f"pairs: positive {len(pos)}, hard negative {len(hard)}, random negative {len(rand)}")

    def feats(pairs):
        rows = []
        for s, i in pairs:
            A, B = S1.loc[s], other.loc[i]
            ra_n, rb_n = raw_toks(A.business_name), raw_toks(B.business_name)
            ca_n, cb_n = set(A.name_core.split()), set(B.name_core.split())
            pa_n, pb_n = set(A.name_phon.split()), set(B.name_phon.split())
            ra_a, rb_a = raw_toks(A.business_address), raw_toks(B.business_address)
            ca_a = set(A.addr_words.split()) | set(A.addr_nums.split())
            cb_a = set(B.addr_words.split()) | set(B.addr_nums.split())
            name_clean = max(jac(ca_n, cb_n), jac(pa_n, pb_n),
                             1.0 if A.name_concat and A.name_concat == B.name_concat else 0.0,
                             1.0 if len(B.name_core) >= 2 and B.name_type == "acronym"
                             and B.name_core == A.name_acronym else 0.0)
            rows.append({
                "country": A.country, "src": B.src, "btype": B.name_type,
                "name_raw_jac": jac(ra_n, rb_n), "name_clean_jac": name_clean,
                "name_raw_fuzz": fuzz.token_set_ratio(fold(A.business_name), fold(B.business_name)),
                "name_clean_fuzz": fuzz.token_set_ratio(A.name_core, B.name_core),
                "addr_raw_jac": jac(ra_a, rb_a), "addr_clean_jac": jac(ca_a, cb_a),
                "addr_raw_fuzz": fuzz.token_set_ratio(fold(A.business_address), fold(B.business_address)),
                "addr_clean_fuzz": fuzz.token_set_ratio(A.addr_words + " " + A.addr_nums,
                                                        B.addr_words + " " + B.addr_nums),
                "state_eq": (A.state == B.state) if (A.state and B.state) else np.nan,
                "house_eq": (A.house_no == B.house_no) if (A.house_no and B.house_no) else np.nan,
                "nums_overlap": bool(set(A.addr_nums.split()) & set(B.addr_nums.split()))
                if (A.addr_nums and B.addr_nums) else np.nan,
                "city_eq": (A.city == B.city) if (A.city and B.city) else np.nan,
                "b_missing": B.addr_missing,
            })
        return pd.DataFrame(rows)

    P, H, Rn = feats(pos), feats(hard), feats(rand)

    def auc(col, neg, mask_p=None, mask_n=None):
        p = P[col] if mask_p is None else P.loc[mask_p, col]
        n = neg[col] if mask_n is None else neg.loc[mask_n, col]
        if len(p) == 0 or len(n) == 0:
            return float("nan")
        return roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p.values, n.values])

    print("\n=== AUC positives vs HARD negatives (raw -> clean) ===")
    for c in ["US", "India"]:
        mp, mn = P.country == c, H.country == c
        for f in ["name_%s_jac", "name_%s_fuzz", "addr_%s_jac", "addr_%s_fuzz"]:
            print(f"  {c:6s} {f % '*':14s} raw {auc(f % 'raw', H, mp, mn):.4f} -> clean {auc(f % 'clean', H, mp, mn):.4f}")
    print("\n=== AUC positives vs RANDOM negatives (raw -> clean) ===")
    for f in ["name_%s_jac", "name_%s_fuzz", "addr_%s_jac", "addr_%s_fuzz"]:
        print(f"  {f % '*':14s} raw {auc(f % 'raw', Rn):.4f} -> clean {auc(f % 'clean', Rn):.4f}")

    print("\n=== name similarity by record type of the S2/S3 side (positives, mean) ===")
    print(P.groupby(["country", "btype"])[["name_raw_jac", "name_clean_jac"]].agg(["mean", "count"]).round(3).to_string())

    print("\n=== share of positives with ANY shared name token / phonetic / concat match ===")
    print(f"  raw {(P.name_raw_jac > 0).mean():.4f} -> clean {(P.name_clean_jac > 0).mean():.4f}")
    print("\n=== component agreement: positives vs hard negatives ===")
    for col in ["state_eq", "house_eq", "nums_overlap", "city_eq"]:
        print(f"  {col:13s} pos {P[col].mean():.4f} (defined {P[col].notna().mean():.3f})   "
              f"hard-neg {H[col].mean():.4f}   random-neg {Rn[col].mean():.4f}")
    print(P.groupby(["country", "src"])[["state_eq", "house_eq", "nums_overlap", "city_eq"]].mean().round(4).to_string())

    # ---------------- S1 stability (Source 1 is clean: fixes should rarely fire) ----------------
    s1n = df[df.src == "S1"]
    changed = []
    for raw, core, legal in zip(s1n.business_name, s1n.name_core, s1n.name_legal):
        rt = [t for t in re.findall(r"[a-z0-9]+", fold(raw).replace("'", ""))]
        ct = set(core.split()) | set(legal.split())
        extra = [t for t in core.split() if t not in rt and not any(t.startswith(x) for x in rt)]
        if extra:
            changed.append((raw, core))
    print(f"\n=== S1 names where cleaning introduced a token not in the raw text: {len(changed)} / {len(s1n)}")
    for x in changed[:25]:
        print("   ", x)

    # ---------------- native fallback accuracy (held-out: pretend the lexicon is missing) ----------------
    # same rule as NameNormalizer._native_token: map only when the phonetic key has >= 5 consonants
    lex = res["native_lex"]
    mapped = mapped_ok = phon_ok = 0
    wrong = []
    for nat, eng in lex.items():
        lat = transliterate_token(nat)
        lat = re.sub(r"a$", "", lat) if len(lat) > 3 else lat
        key = phon(lat)
        guess = res["eng_phon"].get(key, lat) if len(key) >= 5 else lat
        if guess != lat:
            mapped += 1
            if guess == eng or phon(guess) == phon(eng):
                mapped_ok += 1
            elif len(wrong) < 20:
                wrong.append((nat, eng, guess))
        phon_ok += phon(guess) == phon(eng)
    n = len(lex)
    print(f"\n=== native fallback on {n} held-out lexicon words: mapped to an English word {mapped / n:.3f}, "
          f"precision of mappings {mapped_ok / max(1, mapped):.4f}, phonetic key equal to truth {phon_ok / n:.4f}")
    print("    wrong mappings:", wrong)


if __name__ == "__main__":
    main()
