from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make_split_script", default="src/cazy_habench30k/make_homology_split.py")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--thresholds", default="0.30,0.20,0.15,0.10")
    parser.add_argument("--coverage", default="0.80")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for th in args.thresholds.split(","):
        split_out = out / f"id{int(float(th) * 100)}_cov{int(float(args.coverage) * 100)}"
        cmd = [
            "python", args.make_split_script,
            "--dataset_dir", args.dataset_dir,
            "--identity", th,
            "--coverage", args.coverage,
            "--out", str(split_out),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
