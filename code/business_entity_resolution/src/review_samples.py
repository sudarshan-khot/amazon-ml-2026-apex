"""Print random raw -> cleaned records for every split x source x country (manual review)."""
import csv
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_cleaning import NameNormalizer, NameResources, normalize_address  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
res = pickle.load(open(os.path.join(HERE, "..", "artifacts", "resources.pkl"), "rb"))
NN = NameNormalizer(NameResources(res["vocab"], res["native_lex"], res["eng_phon"]))

for split in ("train", "test"):
    for k in "123":
        df = pd.read_csv(os.path.join(DATA, split, f"{split}_source{k}.tsv"), sep="\t", dtype=str,
                         keep_default_na=False, quoting=csv.QUOTE_NONE, nrows=300000)
        for c, g in df.groupby("country"):
            print(f"\n################ {split} S{k} {c}")
            for _, r in g.sample(N, random_state=11).iterrows():
                n = NN.normalize(r.business_name, c)
                a = normalize_address(r.business_address, c)
                print(f"  NAME {r.business_name!r}\n     -> core={n['name_core']!r} legal={n['name_legal']!r} "
                      f"type={n['name_type']} brand={n['name_brand']!r} phon={n['name_phon']!r}")
                print(f"  ADDR {r.business_address!r}\n     -> words={a['addr_words']!r} nums={a['addr_nums']!r} "
                      f"house={a['house_no']!r} unit={a['unit']!r} city={a['city']!r} state={a['state']!r}")
