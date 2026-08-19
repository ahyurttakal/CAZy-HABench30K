from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def sample_k_per_family(labels: pd.DataFrame, family_col: str, k: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _, grp in labels.groupby(family_col):
        n = min(k, len(grp))
        rows.append(grp.sample(n=n, random_state=int(rng.integers(0, 1_000_000))))
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Create low-data supervised train splits.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--family_col", default="family")
    parser.add_argument("--id_col", default="protein_id")
    parser.add_argument("--ks", default="1,5,10,20")
    parser.add_argument("--seeds", default="1,7,42")
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for k in map(int, args.ks.split(",")):
        for seed in map(int, args.seeds.split(",")):
            sub = sample_k_per_family(labels, args.family_col, k, seed)
            split_dir = out / f"k{k}_seed{seed}"
            split_dir.mkdir(parents=True, exist_ok=True)
            sub[[args.id_col]].to_csv(split_dir / "train.csv", index=False)


if __name__ == "__main__":
    main()
