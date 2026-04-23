from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .losses import focal_loss
from .metrics import compute_metrics, find_best_threshold
from .models import build_model


@dataclass
class TrainResult:
    model_name: str
    metrics: Dict[str, float]
    history: List[Dict[str, float]]
    model_state: Dict[str, torch.Tensor]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compute_class_weights(data: Data, device: torch.device) -> torch.Tensor:
    y_train = data.y[data.train_mask]
    pos = int((y_train == 1).sum().item())
    neg = int((y_train == 0).sum().item())
    total = max(1, pos + neg)

    # Inverse frequency with 2 classes.
    w_neg = total / max(1, 2 * neg)
    w_pos = total / max(1, 2 * pos)
    return torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)


def _predict_prob(model: torch.nn.Module, data: Data) -> torch.Tensor:
    logits = model(data.x, data.edge_index)
    return torch.softmax(logits, dim=1)[:, 1]


def _evaluate_mask(model: torch.nn.Module, data: Data, mask: torch.Tensor, threshold: float) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        probs = _predict_prob(model, data)
    y_true = data.y[mask].cpu().numpy()
    y_prob = probs[mask].cpu().numpy()
    return compute_metrics(y_true, y_prob, threshold=threshold).to_dict()


def train_gnn(
    model_name: str,
    data: Data,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    heads: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-4,
    epochs: int = 60,
    patience: int = 12,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    threshold: float = 0.5,
    grad_clip: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
) -> TrainResult:
    set_seed(seed)

    torch_device = torch.device(device)
    data = data.to(torch_device)

    model = build_model(
        name=model_name,
        in_dim=data.x.shape[1],
        hidden_dim=hidden_dim,
        out_dim=2,
        dropout=dropout,
        heads=heads,
    ).to(torch_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    class_weights = _compute_class_weights(data, device=torch_device)

    best_val = -float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        train_logits = logits[data.train_mask]
        train_y = data.y[data.train_mask]

        if use_focal_loss:
            loss = focal_loss(
                train_logits,
                train_y,
                gamma=focal_gamma,
                class_weights=class_weights,
            )
        else:
            loss = F.cross_entropy(train_logits, train_y, weight=class_weights)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        train_metrics = _evaluate_mask(model, data, data.train_mask, threshold)
        model.eval()
        with torch.no_grad():
            val_probs_all = _predict_prob(model, data)
        y_val = data.y[data.val_mask].cpu().numpy()
        y_val_prob = val_probs_all[data.val_mask].cpu().numpy()
        val_metrics_default = compute_metrics(y_true=y_val, y_prob=y_val_prob, threshold=threshold).to_dict()
        tuned_threshold_epoch, val_metrics_tuned = find_best_threshold(y_true=y_val, y_prob=y_val_prob, recall_floor=0.50)
        val_metrics = val_metrics_tuned.to_dict()

        epoch_row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "train_pr_auc": train_metrics["pr_auc"],
            "val_pr_auc": val_metrics_default["pr_auc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_illicit_recall": val_metrics["illicit_recall"],
            "val_fpr": val_metrics["false_positive_rate"],
            "val_selected_threshold": float(tuned_threshold_epoch),
        }
        history.append(epoch_row)

        val_pr_auc = val_metrics_default["pr_auc"]
        if np.isnan(val_pr_auc):
            val_pr_auc = -float("inf")
        val_score = 0.70 * val_pr_auc + 0.30 * val_metrics["macro_f1"]

        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = _predict_prob(model, data)

    y_val = data.y[data.val_mask].cpu().numpy()
    y_val_prob = probs[data.val_mask].cpu().numpy()
    if y_val.shape[0] > 0:
        tuned_threshold, _ = find_best_threshold(y_true=y_val, y_prob=y_val_prob, recall_floor=0.50)
    else:
        tuned_threshold = threshold

    y_test = data.y[data.test_mask].cpu().numpy()
    y_test_prob = probs[data.test_mask].cpu().numpy()
    test_metrics = compute_metrics(y_true=y_test, y_prob=y_test_prob, threshold=float(tuned_threshold)).to_dict()
    test_metrics["selected_threshold"] = float(tuned_threshold)

    return TrainResult(
        model_name=model_name,
        metrics=test_metrics,
        history=history,
        model_state=best_state,
    )


def predict_all_probabilities(
    model_name: str,
    model_state: Dict[str, torch.Tensor],
    data: Data,
    hidden_dim: int,
    dropout: float,
    heads: int,
    device: str = "cpu",
) -> np.ndarray:
    torch_device = torch.device(device)
    model = build_model(
        name=model_name,
        in_dim=data.x.shape[1],
        hidden_dim=hidden_dim,
        out_dim=2,
        dropout=dropout,
        heads=heads,
    ).to(torch_device)
    model.load_state_dict(model_state)
    model.eval()

    data = data.to(torch_device)
    with torch.no_grad():
        probs = _predict_prob(model, data)
    return probs.cpu().numpy()
