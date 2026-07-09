# src2: HierESM reproducibility package

This folder contains the second manuscript code. Keep the original `src/` folder for Paper 1.

## Main commands

```bash
pip install -r src2/requirements-paper2.txt
```

Train HierESM:

```bash
python -m src2.hieresm.train --config src2/configs/hieresm_default.yaml --data_dir data/CAZy-HABench30K --out_dir src2/results/main_benchmark --seed 1
```

Evaluate predictions:

```bash
python -m src2.hieresm.evaluate --predictions src2/results/main_benchmark/predictions_seed1.csv --out src2/results/main_benchmark/metrics_seed1.json
```

Create low-data splits:

```bash
python -m src2.hieresm.analysis.run_low_data --labels data/CAZy-HABench30K/labels.csv --out_dir src2/results/low_data
```

Compute leakage overestimation:

```bash
python -m src2.hieresm.analysis.run_leakage_analysis --cnn_homology 0.628 --cnn_random 0.687 --hieresm_homology 0.877 --hieresm_random 0.909 --out src2/results/leakage/leakage_summary.json
```
