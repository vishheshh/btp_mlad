"""
Utility functions for comprehensive attack evaluation runs.

Currently the evaluation script only needs hour-level metrics, but this module
keeps the logic centralized so other benchmarking scripts can import it without
duplicating confusion-matrix math.
"""

from typing import Dict, Sequence

import numpy as np


def calculate_hour_level_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Dict[str, float]:
    """
    Calculate confusion-matrix metrics at the hour level.

    Args:
        y_true: Iterable of ground-truth labels (1 = attack hour, 0 = normal).
        y_pred: Iterable of predicted labels with the same length.

    Returns:
        Dictionary containing tp, tn, fp, fn, precision, recall, f1, fpr, accuracy.
    """

    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "accuracy": accuracy,
    }


