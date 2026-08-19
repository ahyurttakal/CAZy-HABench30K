# src2: HierESM code for Paper 2

This folder contains the code for the second manuscript:

**Hierarchy-Constrained Protein Language Models for Homology-Aware CAZy Enzyme Family Classification**

It is intentionally separated from the original `src/` folder, which should remain reserved for the first ESM-2+Mean+MTL manuscript.

## Contents

- `hieresm/model.py`: HierESM architecture
- `hieresm/train.py`: training entry point
- `hieresm/evaluate.py`: prediction/metric evaluation
- `hieresm/data.py`: dataset and split loaders
- `hieresm/metrics.py`: Macro-F1, MCC, balanced accuracy, ECE
- `hieresm/baselines/`: HBI and HMMER baseline wrappers
- `hieresm/analysis/`: ablation, low-data, threshold, leakage, long-tail and occlusion scripts
- `configs/hieresm_default.yaml`: default Paper 2 configuration
- `scripts/run_main_benchmark.sh`: multi-seed benchmark runner

## Expected data layout

The dataset itself should be downloaded from Mendeley Data and placed locally, for example:

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

## Install Paper 2 dependencies

```bash
pip install -r src2/requirements-paper2.txt
```

## Example training command

```bash
python -m src2.hieresm.train \
  --config src2/configs/hieresm_default.yaml \
  --data_dir data/CAZy-HABench30K \
  --out_dir src2/results/main_benchmark \
  --seed 1
```
