from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Metrics:
    pr_auc: float
    macro_f1: float
    illicit_recall: float
    precision: float
    roc_auc: float
    false_positive_rate: float
    support_licit: int
    support_illicit: int

    def to_dict(self) -> dict:
        return asdict(self)


def _threshold_objective(metrics: Metrics, recall_floor: float) -> float:
    # Balance high minority recall with practical false-positive control.
    penalty = 0.0 if metrics.illicit_recall >= recall_floor else (recall_floor - metrics.illicit_recall) * 2.0
    return (
        metrics.macro_f1
        + 0.35 * metrics.illicit_recall
        - 0.20 * metrics.false_positive_rate
        - penalty
    )


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    recall_floor: float = 0.50,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    steps: int = 181,
) -> tuple[float, Metrics]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    thresholds = np.linspace(min_threshold, max_threshold, steps)
    best_t = 0.5
    best_m = compute_metrics(y_true, y_prob, threshold=best_t)
    best_score = _threshold_objective(best_m, recall_floor=recall_floor)

    for thr in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=float(thr))
        score = _threshold_objective(m, recall_floor=recall_floor)
        if score > best_score:
            best_t, best_m, best_score = float(thr), m, score

    return best_t, best_m


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Metrics:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    support_licit = int((y_true == 0).sum())
    support_illicit = int((y_true == 1).sum())

    if len(np.unique(y_true)) > 1:
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
    else:
        pr_auc = float("nan")
        roc_auc = float("nan")

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    illicit_recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / max(1, fp + tn))

    return Metrics(
        pr_auc=pr_auc,
        macro_f1=macro_f1,
        illicit_recall=illicit_recall,
        precision=precision,
        roc_auc=roc_auc,
        false_positive_rate=fpr,
        support_licit=support_licit,
        support_illicit=support_illicit,
    )
