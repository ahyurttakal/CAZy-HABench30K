from __future__ import annotations

import argparse
import pandas as pd

from .metrics import summarize_family_and_class
from .utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pred = pd.read_csv(args.predictions)
    required = {"family_true", "family_pred", "class_true", "class_pred"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")

    metrics = summarize_family_and_class(
        pred["family_true"], pred["family_pred"],
        pred["class_true"], pred["class_pred"],
    )
    save_json(metrics, args.out)
    print(metrics)


if __name__ == "__main__":
    main()
