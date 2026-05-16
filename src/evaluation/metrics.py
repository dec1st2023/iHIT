from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion: np.ndarray


def classification_metrics(y_true, y_pred, num_classes: int) -> ClassificationMetrics:
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, guessed in zip(true, pred):
        confusion[actual, guessed] += 1

    accuracy = float((true == pred).mean()) if len(true) else 0.0
    precision = []
    recall = []
    f1 = []
    for idx in range(num_classes):
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision.append(p)
        recall.append(r)
        f1.append((2 * p * r / (p + r)) if p + r else 0.0)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_macro=float(np.mean(precision)),
        recall_macro=float(np.mean(recall)),
        f1_macro=float(np.mean(f1)),
        confusion=confusion,
    )
