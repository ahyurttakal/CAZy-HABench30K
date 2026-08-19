# Reproducibility Notes

To reproduce the main experiments:

1. Download the CAZy-HABench30K data package from Mendeley Data.
2. Install the Python dependencies listed in `requirements.txt`.
3. Install MMseqs2 if you want to regenerate the homology-aware split.
4. Use the fixed train/validation/test split provided in the data package for model training and evaluation.
5. For first paper, run `src/cazy_habench30k/train_esm2_mtl.py` after updating local paths and hardware-specific arguments.
6. For second paper, run `src2/scripts/run_main_benchmark.py` after updating local paths and hardware-specific arguments.

