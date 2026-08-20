"""subsample_retrieval_candidates.py

Randomly subsample N unique `spec` entries from a cands_df_*.tsv retrieval
candidate file, keeping each sampled spec's FULL candidate pool intact (all
~50 rows per spec) -- Top-1 ranking needs the complete pool for a spectrum,
so we can't subsample individual candidate rows.

Usage:
    .venv/bin/python subsample_retrieval_candidates.py \
        --in-tsv <cands_df_split_1_50.tsv> \
        --out-tsv <cands_df_split_1_50_sample300.tsv> \
        --n-specs 300 --seed 42
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-tsv", required=True)
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--n-specs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.in_tsv, sep="\t")
    unique_specs = df["spec"].unique()
    print(f"Total unique specs in {args.in_tsv}: {len(unique_specs)}")

    n = min(args.n_specs, len(unique_specs))
    sampled_specs = pd.Series(unique_specs).sample(n=n, random_state=args.seed)
    sub = df[df["spec"].isin(set(sampled_specs))]

    sub.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"Sampled {n} specs -> {len(sub)} candidate rows")
    print(f"Wrote {args.out_tsv}")


if __name__ == "__main__":
    main()
