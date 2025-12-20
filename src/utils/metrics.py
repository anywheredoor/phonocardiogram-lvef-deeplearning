#!/usr/bin/env python3
"""
Common evaluation metrics for binary LVEF classification.

Primary metric is F1 for the positive class (label=1, EF <= 40).
Also reports AUROC, AUPRC, accuracy, sensitivity, and specificity.
"""

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def sigmoid_probs(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid to avoid exp overflow warnings."""
    x = np.asarray(x)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out = np.empty_like(x, dtype=np.float64)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)
    return out


def compute_binary_metrics(
    logits, labels, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute key binary classification metrics given logits and integer labels.
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels).astype(int)

    probs = sigmoid_probs(logits)
    preds = (probs >= threshold).astype(int)

    # F1 for the positive class (label=1)
    f1_pos = f1_score(labels, preds, pos_label=1, zero_division=0)

    acc = accuracy_score(labels, preds)

    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    except ValueError:
        tn = fp = fn = tp = 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUROC/AUPRC only meaningful if both classes are present
    if len(np.unique(labels)) == 2:
        try:
            auroc = roc_auc_score(labels, probs)
        except ValueError:
            auroc = float("nan")
        try:
            auprc = average_precision_score(labels, probs)
        except ValueError:
            auprc = float("nan")
    else:
        auroc = float("nan")
        auprc = float("nan")

    return {
        "f1_pos": float(f1_pos),
        "accuracy": float(acc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auroc": float(auroc),
        "auprc": float(auprc),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """Compact string formatter for logging."""
    return (
        f"F1_pos={metrics['f1_pos']:.4f}, "
        f"AUROC={metrics['auroc']:.4f}, "
        f"AUPRC={metrics['auprc']:.4f}, "
        f"Acc={metrics['accuracy']:.4f}, "
        f"Sens={metrics['sensitivity']:.4f}, "
        f"Spec={metrics['specificity']:.4f}"
    )


def tune_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: Iterable[float] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Search over thresholds to maximise F1 for the positive class.

    Args:
        logits: Array of model logits.
        labels: Array of integer labels (0/1).
        thresholds: Iterable of thresholds to try. If None, uses a default
            grid from 0.05 to 0.95 (step 0.05).

    Returns:
        best_threshold: Threshold achieving highest F1_pos (tie broken by AUROC).
        best_metrics: Metrics computed at best_threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    best_thr = None
    best_metrics = None
    best_f1 = -np.inf
    best_auroc = -np.inf

    for thr in thresholds:
        metrics = compute_binary_metrics(logits, labels, threshold=thr)
        f1 = metrics["f1_pos"]
        auroc = metrics["auroc"]
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and auroc > best_auroc):
            best_f1 = f1
            best_auroc = auroc
            best_thr = float(thr)
            best_metrics = metrics

    return best_thr, best_metrics
