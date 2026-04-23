from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.ensemble import HistGradientBoostingClassifier

from .metrics import compute_metrics, find_best_threshold

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


@dataclass
class BaselineResult:
    model_name: str
    metrics: Dict[str, float]


def run_tabular_baseline(data: Data, threshold: float = 0.5, seed: int = 42) -> BaselineResult:
    x = data.x.cpu().numpy()
    y = data.y.cpu().numpy()

    train_mask = data.train_mask.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    pos = max(1, int((y_train == 1).sum()))
    neg = max(1, int((y_train == 0).sum()))

    if HAS_XGBOOST:
        model_name = "xgboost"
        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=seed,
            scale_pos_weight=float(neg / pos),
            n_jobs=4,
        )
    else:
        model_name = "hist_gradient_boosting"
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=8,
            max_iter=300,
            random_state=seed,
        )

    model.fit(x_train, y_train)

    def _predict_prob(x_input: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x_input)[:, 1]
        raw = model.decision_function(x_input)
        return 1.0 / (1.0 + np.exp(-raw))

    y_val_prob = _predict_prob(x_val)
    if y_val.shape[0] > 0:
        tuned_threshold, _ = find_best_threshold(y_true=y_val, y_prob=y_val_prob, recall_floor=0.50)
    else:
        tuned_threshold = threshold

    y_prob = _predict_prob(x_test)
    metrics = compute_metrics(y_true=y_test, y_prob=y_prob, threshold=float(tuned_threshold)).to_dict()
    metrics["selected_threshold"] = float(tuned_threshold)

    return BaselineResult(model_name=model_name, metrics=metrics)
