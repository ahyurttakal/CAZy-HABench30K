from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def make_windows(seq: str, window: int, stride: int):
    for start in range(0, max(1, len(seq) - window + 1), stride):
        end = min(len(seq), start + window)
        yield start, end, seq[:start] + ("X" * (end - start)) + seq[end:]


def main():
    parser = argparse.ArgumentParser(description="Create occlusion windows for selected proteins.")
    parser.add_argument("--cases_csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--id_col", default="protein_id")
    parser.add_argument("--seq_col", default="sequence")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    cases = pd.read_csv(args.cases_csv)
    rows = []
    for _, row in cases.iterrows():
        for start, end, masked in make_windows(str(row[args.seq_col]), args.window, args.stride):
            rows.append({
                "protein_id": row[args.id_col],
                "start": start + 1,
                "end": end,
                "masked_sequence": masked,
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
