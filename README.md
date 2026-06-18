# CAZy-HABench30K

Reproducibility code for **CAZy-HABench30K**, a homology-aware benchmark for CAZy enzyme family classification.

The repository contains the scripts used to construct the benchmark dataset, generate MMseqs2-based cluster-disjoint splits, and train/evaluate the ESM-2+Mean+MTL model.

The dataset itself is deposited separately in Mendeley Data.

## Repository structure

```text
CAZy-HABench30K/
├── src/
│   └── cazy_habench30k/
│       ├── build_dataset.py
│       ├── make_homology_split.py
│       └── train_esm2_mtl.py
├── docs/
│   ├── DATA_AVAILABILITY.md
│   └── REPRODUCIBILITY.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Main scripts

- `src/cazy_habench30k/build_dataset.py`  
  Builds the initial UniProt–CAZy 30K benchmark dataset and generates sequence, label, family-count, and class-coverage files.

- `src/cazy_habench30k/make_homology_split.py`  
  Creates the MMseqs2-based homology-aware train/validation/test split.

- `src/cazy_habench30k/train_esm2_mtl.py`  
  Trains and evaluates the ESM-2+Mean+MTL model and supported baselines under the fixed benchmark split.

## Data

The data are available from Mendeley Data:

```text
Mendeley Data DOI/link: https://doi.org/10.17632/m9r9pb39jw
```

The data package includes FASTA sequences, family/class labels, MMseqs2 cluster assignments, fixed train/validation/test splits, prediction outputs, and result tables.

## Installation

```bash
pip install -r requirements.txt
```

MMseqs2 is required for homology-aware split generation and should be installed separately.

## Basic workflow

### 1. Build the initial benchmark dataset

```bash
python src/cazy_habench30k/build_dataset.py \
    --out UniProt-CAZy-30k \
    --target_total 30000 \
    --min_per_family 100 \
    --ce_min_per_family 20 \
    --aa_min_per_family 20 \
    --max_per_family 600 \
    --priority_classes CE,AA
```

### 2. Generate or verify the homology-aware split

```bash
python src/cazy_habench30k/make_homology_split.py \
    --dataset_dir UniProt-CAZy-30k \
    --out splits_id20_cov60
```

The fixed split used in the manuscript is also provided in the Mendeley Data package.

### 3. Train and evaluate the model

```bash
python src/cazy_habench30k/train_esm2_mtl.py
```

Please update local file paths and training arguments according to your computing environment.

## Citation

If you use this repository or the CAZy-HABench30K dataset, please cite the associated manuscript and the Mendeley Data record.

```text
Yurttakal, Ahmet Haşim; Erbay, Hasan (2026), “CAZy-HABench30K: A Homology-Aware Benchmark Dataset for CAZy Enzyme Family Classification”, Mendeley Data, V1, doi: 10.17632/m9r9pb39jw
```

## License

This code is released under the MIT License.
