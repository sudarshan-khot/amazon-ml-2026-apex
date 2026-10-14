"""Final checks on the cleaned TSV files written by data_cleaning.py.

1. every output file parses with the documented reader: right columns, same rows / order as input
2. random records re-cleaned from the raw data match the file contents exactly (round trip)
3. state_inf only fills missing states and never overwrites an observed state
4. values like 'nan' / 'null' survive as text (no NA conversion)
Usage: python src/check_cleaned_files.py --data ../../dataset
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_cleaning import OUT_COLUMNS, ROOT, clean_frame, load_name_normalizer, read_tsv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--clean", default=os.path.join(ROOT, "artifacts", "cleaned_dataset"))
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    nn = load_name_normalizer(os.path.join(ROOT, "artifacts", "resources.pkl"))
    ok = True
    for split in ("train", "test"):
        for k in (1, 2, 3):
            raw = read_tsv(os.path.join(a.data, split, f"{split}_source{k}.tsv"))
            cl = read_tsv(os.path.join(a.clean, split, f"{split}_source{k}.tsv"))
            msgs = []
            # 1) structure
            if list(cl.columns) != OUT_COLUMNS:
                msgs.append(f"columns {list(cl.columns)}")
            if len(cl) != len(raw) or not (cl.entity_id.values == raw.entity_id.values).all():
                msgs.append("row count / order differs from input")
            if not (cl.country.values == raw.country.values).all():
                msgs.append("country column differs from input")
            # 2) round trip on a random sample
            idx = raw.sample(min(a.n, len(raw)), random_state=7).index
            fresh = clean_frame(raw.loc[idx], nn)
            stored = cl.loc[idx].reset_index(drop=True)
            cols = [c for c in OUT_COLUMNS if c != "state_inf"]
            mism = (fresh[cols].values != stored[cols].values).any(axis=1).sum()
            if mism:
                msgs.append(f"{mism} sampled rows differ from a fresh cleaning")
            # 3) state_inf
            overwritten = ((cl.state != "") & (cl.state_inf != cl.state)).sum()
            if overwritten:
                msgs.append(f"state_inf overwrote {overwritten} observed states")
            filled = ((cl.state == "") & (cl.state_inf != "")).sum()
            # 4) NA-like strings kept as text
            nan_like = cl.name_core.isin(["nan", "null", "na", "none"]).sum()
            status = "OK" if not msgs else "FAIL: " + "; ".join(msgs)
            ok &= not msgs
            print(f"{split} source{k}: {len(cl):,} rows | round-trip sample {len(idx):,} | "
                  f"state_inf filled {filled:,} | NA-like names kept as text {nan_like} | {status}", flush=True)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
