from __future__ import annotations

from typing import Dict, Any
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score, accuracy_score


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def classification_metrics(y_true, y_pred, y_prob=None, n_bins: int = 15) -> Dict[str, Any]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_prob is not None:
        out["ece"] = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    return out


def summarize_family_and_class(
    family_true,
    family_pred,
    class_true,
    class_pred,
    family_prob=None,
    class_prob=None,
    n_bins: int = 15,
) -> Dict[str, Any]:
    fam = classification_metrics(family_true, family_pred, family_prob, n_bins)
    cls = classification_metrics(class_true, class_pred, class_prob, n_bins)
    return {
        "family_macro_f1": fam["macro_f1"],
        "family_weighted_f1": fam["weighted_f1"],
        "family_mcc": fam["mcc"],
        "family_balanced_accuracy": fam["balanced_accuracy"],
        "family_ece": fam.get("ece"),
        "class_macro_f1": cls["macro_f1"],
        "class_mcc": cls["mcc"],
        "class_ece": cls.get("ece"),
    }
