from __future__ import annotations

import argparse
import json
from pathlib import Path


def relative_overestimation(random_macro_f1: float, homology_macro_f1: float) -> float:
    return 100.0 * (random_macro_f1 - homology_macro_f1) / homology_macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn_homology", type=float, required=True)
    parser.add_argument("--cnn_random", type=float, required=True)
    parser.add_argument("--hieresm_homology", type=float, required=True)
    parser.add_argument("--hieresm_random", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    summary = {
        "CNN": {
            "homology_macro_f1": args.cnn_homology,
            "random_macro_f1": args.cnn_random,
            "relative_overestimation_percent": relative_overestimation(args.cnn_random, args.cnn_homology),
        },
        "HierESM": {
            "homology_macro_f1": args.hieresm_homology,
            "random_macro_f1": args.hieresm_random,
            "relative_overestimation_percent": relative_overestimation(args.hieresm_random, args.hieresm_homology),
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
