from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd):
    print(" ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="MMseqs2 nearest-neighbour HBI baseline wrapper.")
    parser.add_argument("--train_fasta", required=True)
    parser.add_argument("--test_fasta", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sensitivity", default="7.5")
    parser.add_argument("--coverage", default="0.5")
    parser.add_argument("--cov_mode", default="0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_tsv = out_dir / "mmseqs_hbi_hits.tsv"
    tmp = out_dir / "tmp_mmseqs"

    run([
        "mmseqs", "easy-search",
        args.test_fasta,
        args.train_fasta,
        str(result_tsv),
        str(tmp),
        "-s", str(args.sensitivity),
        "-c", str(args.coverage),
        "--cov-mode", str(args.cov_mode),
        "--format-output", "query,target,pident,alnlen,evalue,bits",
    ])

    print(f"MMseqs2 hits written to {result_tsv}")


if __name__ == "__main__":
    main()
