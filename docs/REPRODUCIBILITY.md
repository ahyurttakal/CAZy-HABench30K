# Reproducibility Notes

To reproduce the main experiments:

1. Download the CAZy-HABench30K data package from Mendeley Data.
2. Install the Python dependencies listed in `requirements.txt`.
3. Install MMseqs2 if you want to regenerate the homology-aware split.
4. Use the fixed train/validation/test split provided in the data package for model training and evaluation.
5. Run `src/cazy_habench30k/train_esm2_mtl.py` after updating local paths and hardware-specific arguments.

The fixed benchmark split contains:

```text
Train: 21,000 sequences
Validation: 4,500 sequences
Test: 4,500 sequences
Families: 60
Classes: 6
```
