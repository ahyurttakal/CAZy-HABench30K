# CAZy-HABench30K

Reproducibility repository for **CAZy-HABench30K**, a homology-aware benchmark and modelling framework for CAZy enzyme family classification.

This repository contains code for below manuscript tracks:

Yurttakal A.H. & Erbay H. *Homology-aware benchmarking of deep learning models for CAZy enzyme family classification*. **Discover Artificial Intelligence** (2026).  
   DOI: https://doi.org/10.1007/s44163-026-01861-5

The repository is organized so that the original Paper code remains available under `src/`.

---

## Repository structure

```text
CAZy-HABench30K/
├── src/                         
│   └── cazy_habench30k/
│       ├── __init__.py
│       ├── build_dataset.py
│       ├── make_homology_split.py
│       └── train_esm2_mtl.py
├── docs/
│   ├── DATA_AVAILABILITY.md
│   └── REPRODUCIBILITY.md
├── README.md
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Paper: ESM-2+Mean+MTL benchmark study

The manuscript introduced the CAZy-HABench30K benchmark workflow, including dataset construction, MMseqs2-based cluster-disjoint splitting, and ESM-2+Mean+MTL model training/evaluation.

```text
Yurttakal A.H. & Erbay H.
Homology-aware benchmarking of deep learning models for CAZy enzyme family classification.
Discover Artificial Intelligence (2026).
DOI: https://doi.org/10.1007/s44163-026-01861-5
```

**Code location**

```text
src/
```

**Main scripts**

- `src/cazy_habench30k/build_dataset.py`  
  Builds the UniProt–CAZy 30K benchmark dataset and generates sequence, label, family-count, and class-coverage files.

- `src/cazy_habench30k/make_homology_split.py`  
  Creates the MMseqs2-based homology-aware train/validation/test split.

- `src/cazy_habench30k/train_esm2_mtl.py`  
  Trains and evaluates the ESM-2+Mean+MTL model and supported baselines under the fixed benchmark split.

**Example workflow**

```bash
python src/cazy_habench30k/build_dataset.py \
    --out UniProt-CAZy-30k \
    --target_total 30000 \
    --min_per_family 100 \
    --ce_min_per_family 20 \
    --aa_min_per_family 20 \
    --max_per_family 600 \
    --priority_classes CE,AA

python src/cazy_habench30k/make_homology_split.py \
    --dataset_dir UniProt-CAZy-30k \
    --out splits_id20_cov60

python src/cazy_habench30k/train_esm2_mtl.py
```

---

## Data

The dataset itself is deposited separately in Mendeley Data.

```text
CAZy-HABench30K: A Homology-Aware Benchmark Dataset for CAZy Enzyme Family Classification
Mendeley Data
DOI: 10.17632/m9r9pb39jw.2
Link: https://data.mendeley.com/datasets/m9r9pb39jw/2
```

---

## Installation

A single top-level `requirements.txt` is provided for both manuscript tracks.

```bash
pip install -r requirements.txt
```

---

## Reproducibility notes

The repository is intended to support reproducibility through:

- source code
- configuration files
- fixed split files or Mendeley Data reference
- prediction outputs
- result tables
- analysis scripts
- plotting scripts

Large trained checkpoints are not required if fixed splits, prediction outputs, configuration files, and analysis scripts are provided. If checkpoints are uploaded later, Git LFS is recommended.

---

## Citation

If you use the dataset, please cite:

```text
Yurttakal A.H. & Erbay H.
Homology-aware benchmarking of deep learning models for CAZy enzyme family classification.
Discover Artificial Intelligence (2026).
https://doi.org/10.1007/s44163-026-01861-5
```

```

```text
Yurttakal A.H. & Erbay H.
CAZy-HABench30K: A Homology-Aware Benchmark Dataset for CAZy Enzyme Family Classification.
Mendeley Data, V2, (2026).
https://doi.org/10.17632/m9r9pb39jw.2
```

---

## License

This code is released under the MIT License.
