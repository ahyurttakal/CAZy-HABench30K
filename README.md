# CAZy-HABench30K

Reproducibility repository for **CAZy-HABench30K**, a homology-aware benchmark and modelling framework for CAZy enzyme family classification.

This repository contains code for two related manuscript tracks:

1. **Paper 1 — accepted/published**  
   Yurttakal A.H. & Erbay H. *Homology-aware benchmarking of deep learning models for CAZy enzyme family classification*. **Discover Artificial Intelligence** (2026).  
   DOI: https://doi.org/10.1007/s44163-026-01861-5

2. **Paper 2 — under review**  
   *Hierarchy-Constrained Protein Language Models for Homology-Aware CAZy Enzyme Family Classification*.  
   This manuscript introduces **HierESM**, a hierarchy-constrained protein language model for homology-aware CAZy family classification.

The repository is organized so that the original Paper 1 code remains available under `src/`, while the Paper 2 code is provided separately under `src2/`.

---

## Repository structure

```text
CAZy-HABench30K/
├── src/                         # Paper 1: ESM-2+Mean+MTL accepted study
│   └── cazy_habench30k/
│       ├── __init__.py
│       ├── build_dataset.py
│       ├── make_homology_split.py
│       └── train_esm2_mtl.py
│
├── src2/                        # Paper 2: HierESM under-review study
│   ├── configs/
│   │   └── hieresm_default.yaml
│   ├── hieresm/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── utils.py
│   │   ├── baselines/
│   │   │   ├── __init__.py
│   │   │   ├── run_hbi_baseline.py
│   │   │   └── run_hmmer_baseline.py
│   │   └── analysis/
│   │       ├── __init__.py
│   │       ├── run_ablation.py
│   │       ├── run_leakage_analysis.py
│   │       ├── run_long_tail.py
│   │       ├── run_low_data.py
│   │       ├── run_occlusion.py
│   │       └── run_threshold_sensitivity.py
│   └── scripts/
│       └── run_main_benchmark.sh
│
├── docs/
│   ├── DATA_AVAILABILITY.md
│   └── REPRODUCIBILITY.md
├── README.md
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Paper 1: ESM-2+Mean+MTL benchmark study

The first manuscript introduced the CAZy-HABench30K benchmark workflow, including dataset construction, MMseqs2-based cluster-disjoint splitting, and ESM-2+Mean+MTL model training/evaluation.

**Published article**

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

## Paper 2: HierESM hierarchy-constrained model

The second manuscript introduces **HierESM**, a hierarchy-constrained protein language modelling framework for homology-aware CAZy family classification.

HierESM combines:

- ESM-2 35M frozen backbone
- LoRA-based parameter-efficient adaptation
- learnable attention pooling
- Hierarchical Conditional Head for class–family consistency
- LDAM-DRW for long-tail-aware learning

**Manuscript status**

```text
Under review
```

**Code location**

```text
src2/
```

**Main Paper 2 components**

- `src2/hieresm/model.py`  
  HierESM architecture.

- `src2/hieresm/train.py`  
  Training entry point.

- `src2/hieresm/evaluate.py`  
  Prediction and metric evaluation.

- `src2/hieresm/data.py`  
  Dataset and split loaders.

- `src2/hieresm/metrics.py`  
  Macro-F1, weighted F1, MCC, balanced accuracy, and ECE.

- `src2/hieresm/baselines/`  
  HBI and HMMER baseline wrappers.

- `src2/hieresm/analysis/`  
  Ablation, low-data supervised evaluation, identity-threshold sensitivity, homology leakage, long-tail, and occlusion scripts.

- `src2/configs/hieresm_default.yaml`  
  Default HierESM configuration.

- `src2/scripts/run_main_benchmark.sh`  
  Multi-seed benchmark runner.

**Example HierESM training command**

```bash
python -m src2.hieresm.train \
  --config src2/configs/hieresm_default.yaml \
  --data_dir data/CAZy-HABench30K \
  --out_dir src2/results/main_benchmark \
  --seed 1
```

**Run the multi-seed main benchmark**

```bash
bash src2/scripts/run_main_benchmark.sh data/CAZy-HABench30K src2/results/main_benchmark
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

Expected local layout for Paper 2:

```text
data/CAZy-HABench30K/
├── labels.csv
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

`labels.csv` should contain at least:

```text
protein_id,sequence,family,class
```

---

## Installation

A single top-level `requirements.txt` is provided for both manuscript tracks.

```bash
pip install -r requirements.txt
```

External command-line tools may also be required depending on which analyses are reproduced:

- **MMseqs2** for homology-aware splitting and HBI nearest-neighbour baseline
- **HMMER** for profile-HMM baseline
- **MAFFT** or **Clustal Omega** if family-level multiple-sequence alignments are regenerated before HMMER profile construction

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

If you use the Paper 1 code, please cite:

```text
Yurttakal A.H. & Erbay H.
Homology-aware benchmarking of deep learning models for CAZy enzyme family classification.
Discover Artificial Intelligence (2026).
https://doi.org/10.1007/s44163-026-01861-5
```

If you use the Paper 2 HierESM code, please cite the associated manuscript once it becomes publicly available. Until then, cite this repository and the CAZy-HABench30K dataset.

If you use the dataset, please cite the Mendeley Data record:

```text
Yurttakal, Ahmet Haşim; Erbay, Hasan (2026),
“CAZy-HABench30K: A Homology-Aware Benchmark Dataset for CAZy Enzyme Family Classification”,
Mendeley Data, V2, doi: 10.17632/m9r9pb39jw.2
```

---

## License

This code is released under the MIT License.
