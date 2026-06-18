# CAZy-HABench30K Code

This repository contains the source code used for the CAZy-HABench30K study: a homology-aware benchmark for CAZy enzyme family classification.

The benchmark data are not stored in this repository. The FASTA sequences, labels, MMseqs2 cluster assignments, fixed train/validation/test splits, prediction outputs, and result tables are available from the associated Mendeley Data record.

## Repository structure

```text
CAZy_HABench30K_Code_GitHub/
├── code/
│   ├── build_uniprot_cazy_30k.py
│   ├── create_homology_aware_split.py
│   └── train_esm2_mean_mtl.py
├── revision_scripts/
├── legacy_versions/
├── docs/
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Main scripts

- `code/build_uniprot_cazy_30k.py`: constructs the initial UniProt–CAZy 30K benchmark by retrieving protein sequences and assigning CAZy family/class labels.
- `code/create_homology_aware_split.py`: generates MMseqs2-based homology-aware train/validation/test splits.
- `code/train_esm2_mean_mtl.py`: trains and evaluates the ESM-2+Mean+MTL model under the fixed homology-aware split.

## Data

Please download the dataset from Mendeley Data:

```text
Mendeley Data DOI/link: <insert DOI or URL here>
```

The expected data structure is:

```text
data/
├── sequences/
│   ├── all.fasta
│   ├── train.fasta
│   ├── val.fasta
│   └── test.fasta
├── labels/
│   ├── labels.csv
│   ├── train_labels.csv
│   ├── val_labels.csv
│   └── test_labels.csv
├── splits/
│   ├── mmseqs2_clusters_id20_cov60.tsv
│   └── split_report.json
├── predictions/
└── results/
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

MMseqs2 must also be installed and available in the system path for homology-aware clustering and split generation.

## Example workflow

### 1. Construct the initial dataset

```bash
python code/build_uniprot_cazy_30k.py \
    --out UniProt-CAZy-Q1-30k \
    --target_total 30000 \
    --min_per_family 100 \
    --ce_min_per_family 20 \
    --aa_min_per_family 20 \
    --max_per_family 600 \
    --priority_classes CE,AA
```

### 2. Generate or verify the homology-aware split

Use the MMseqs2-based split script and the final split files provided in the Mendeley Data package. The final benchmark split used in the manuscript contains:

```text
Train: 21,000 sequences
Validation: 4,500 sequences
Test: 4,500 sequences
Families: 60
Classes: 6
```

### 3. Train and evaluate the model

```bash
python code/train_esm2_mean_mtl.py
```

Please check the script arguments and paths before running, as local directory names may need to be adjusted.

## Citation

If you use this code or the CAZy-HABench30K benchmark, please cite the associated manuscript and the Mendeley Data record.

## License

This code is released under the MIT License.
