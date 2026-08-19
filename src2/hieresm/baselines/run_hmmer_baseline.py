from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd):
    print(" ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Training-derived HMMER profile baseline wrapper.")
    parser.add_argument("--family_alignment_dir", required=True)
    parser.add_argument("--test_fasta", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    aln_dir = Path(args.family_alignment_dir)
    out_dir = Path(args.out_dir)
    hmm_dir = out_dir / "hmms"
    hmm_dir.mkdir(parents=True, exist_ok=True)

    hmm_files = []
    for aln in sorted(aln_dir.glob("*.fa*")):
        hmm = hmm_dir / f"{aln.stem}.hmm"
        run(["hmmbuild", str(hmm), str(aln)])
        hmm_files.append(hmm)

    combined = out_dir / "families.hmm"
    with combined.open("wb") as w:
        for hmm in hmm_files:
            w.write(hmm.read_bytes())

    run(["hmmpress", str(combined)])

    tblout = out_dir / "hmmer_tblout.tsv"
    run(["hmmsearch", "--tblout", str(tblout), str(combined), args.test_fasta])

    print(f"HMMER results written to {tblout}")


if __name__ == "__main__":
    main()
