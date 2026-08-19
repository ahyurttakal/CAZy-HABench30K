from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import spearmanr


def bin_family_size(n):
    if n > 200:
        return "head"
    if n > 50:
        return "medium"
    return "tail"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--family_col", default="family")
    parser.add_argument("--family_true_col", default="family_true")
    parser.add_argument("--family_pred_col", default="family_pred")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    pred = pd.read_csv(args.predictions)
    counts = labels[args.family_col].value_counts().rename("train_family_size")

    rows = []
    for fam, grp in pred.groupby(args.family_true_col):
        f1 = f1_score(grp[args.family_true_col], grp[args.family_pred_col], average="macro", zero_division=0)
        size = int(counts.get(fam, 0))
        rows.append({"family": fam, "family_size": size, "bin": bin_family_size(size), "f1": f1})

    out_df = pd.DataFrame(rows)
    summary = out_df.groupby("bin")["f1"].mean().reset_index(name="mean_family_f1")
    rho, p = spearmanr(out_df["family_size"], out_df["f1"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path.with_name(out_path.stem + "_per_family.csv"), index=False)
    summary.to_csv(out_path, index=False)
    print({"spearman_rho": rho, "p_value": p})


if __name__ == "__main__":
    main()
