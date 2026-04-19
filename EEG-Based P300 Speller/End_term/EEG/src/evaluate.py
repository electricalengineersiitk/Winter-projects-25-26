"""
evaluate.py
===========
Cross-validation, ITR calculation, metric reporting, and result plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import math
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from pathlib import Path

log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Information Transfer Rate
# ---------------------------------------------------------------------------

def compute_itr(n_symbols: int,
                accuracy: float,
                trial_duration_s: float) -> float:
    """
    Compute Information Transfer Rate (ITR) in bits/minute.

    Parameters
    ----------
    n_symbols       : vocabulary size (e.g. 36 for 6×6 matrix)
    accuracy        : classification accuracy as a fraction [0, 1]
    trial_duration_s: duration of one character trial in seconds

    Returns
    -------
    itr : float — bits per minute
    """
    P = np.clip(accuracy, 1e-9, 1 - 1e-9)
    N = n_symbols

    if P == 1.0:
        bits_per_trial = math.log2(N)
    elif P <= 1.0 / N:
        bits_per_trial = 0.0
    else:
        bits_per_trial = (
            math.log2(N)
            + P * math.log2(P)
            + (1 - P) * math.log2((1 - P) / (N - 1))
        )

    itr = bits_per_trial * (60.0 / trial_duration_s)
    return round(itr, 4)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(model,
                          X: np.ndarray,
                          y: np.ndarray,
                          k: int = 5,
                          random_state: int = 42) -> dict:
    """
    Stratified k-fold cross-validation.

    Parameters
    ----------
    model : sklearn-compatible estimator
    X     : feature matrix (n_samples, n_features) or (n_samples, n_ch, n_times)
    y     : labels (n_samples,)
    k     : number of folds

    Returns
    -------
    results : dict with mean/std of accuracy, precision, recall, F1
    """
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    scoring = {
        "accuracy":  "accuracy",
        "precision": "precision_macro",
        "recall":    "recall_macro",
        "f1":        "f1_macro",
    }

    log.info("Running %d-fold stratified cross-validation …", k)
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring,
                                return_train_score=False, n_jobs=-1)

    results = {}
    for metric in ["accuracy", "precision", "recall", "f1"]:
        scores = cv_results[f"test_{metric}"]
        results[metric] = {"mean": scores.mean(), "std": scores.std()}
        log.info("  %-12s: %.4f ± %.4f", metric, scores.mean(), scores.std())

    return results


def evaluate_on_test(model,
                     X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray,  y_test: np.ndarray,
                     n_symbols: int = 36,
                     trial_duration_s: float = 2.0) -> dict:
    """
    Train on X_train, evaluate on X_test.
    Returns metrics dict including ITR.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1    = f1_score(y_test, y_pred, average="macro", zero_division=0)
    itr   = compute_itr(n_symbols, acc, trial_duration_s)
    cm    = confusion_matrix(y_test, y_pred)

    log.info("Test accuracy : %.4f", acc)
    log.info("Test precision: %.4f", prec)
    log.info("Test recall   : %.4f", rec)
    log.info("Test F1       : %.4f", f1)
    log.info("ITR           : %.2f bits/min", itr)

    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "itr":       itr,
        "confusion_matrix": cm,
        "y_pred":    y_pred,
        "y_test":    y_test,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_erp(epochs, channel: str = "Cz",
             conditions: dict | None = None,
             save_path: str | None = None):
    """
    Plot grand-average ERP waveforms for target vs. non-target epochs.

    Parameters
    ----------
    epochs     : MNE Epochs object with event labels
    channel    : channel to plot (e.g. 'Cz')
    conditions : dict mapping condition name → event label string
    save_path  : if given, save figure to this path
    """
    import mne
    conditions = conditions or {"Target": "target", "Non-target": "nontarget"}
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"Target": "#e74c3c", "Non-target": "#3498db"}
    for label, event_key in conditions.items():
        try:
            evoked = epochs[event_key].average()
            ch_idx = evoked.ch_names.index(channel)
            times = evoked.times * 1000          # convert to ms
            data = evoked.data[ch_idx] * 1e6     # convert to µV
            ax.plot(times, data, label=label, color=colors.get(label), lw=2)
        except (KeyError, ValueError):
            log.warning("Condition '%s' not found in epochs.", event_key)

    ax.axvline(0, color="k", linestyle="--", lw=1, label="Stimulus onset")
    ax.axhline(0, color="grey", linestyle="-", lw=0.5)
    ax.set_xlabel("Time (ms)", fontsize=13)
    ax.set_ylabel("Amplitude (µV)", fontsize=13)
    ax.set_title(f"Grand-average ERP — Channel {channel}", fontsize=15)
    ax.legend(fontsize=11)
    ax.set_xlim(-200, 800)
    sns.despine(fig)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        log.info("ERP plot saved to %s", save_path)
    return fig


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: list | None = None,
                          title: str = "Confusion Matrix",
                          save_path: str | None = None):
    """
    Plot a styled confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = class_names or [str(i) for i in range(len(cm))]

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title(title, fontsize=15)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        log.info("Confusion matrix saved to %s", save_path)
    return fig


def plot_accuracy_vs_itr(results_list: list[dict],
                         model_names: list[str],
                         save_path: str | None = None):
    """
    Bar charts comparing accuracy and ITR across models.

    Parameters
    ----------
    results_list : list of evaluate_on_test() result dicts
    model_names  : list of model names (same order)
    """
    accs = [r["accuracy"] for r in results_list]
    itrs = [r["itr"]      for r in results_list]
    x    = np.arange(len(model_names))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(x, accs, color="#3498db", edgecolor="white", width=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(model_names)
    axes[0].set_ylabel("Accuracy"); axes[0].set_title("Classification Accuracy")
    axes[0].set_ylim(0, 1.05)
    for xi, v in zip(x, accs):
        axes[0].text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    axes[1].bar(x, itrs, color="#e67e22", edgecolor="white", width=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(model_names)
    axes[1].set_ylabel("ITR (bits/min)"); axes[1].set_title("Information Transfer Rate")
    for xi, v in zip(x, itrs):
        axes[1].text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=10)

    sns.despine(fig)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        log.info("Accuracy vs ITR plot saved to %s", save_path)
    return fig


def save_metrics_csv(results_list: list[dict],
                     model_names: list[str],
                     filepath: str = "results/metrics.csv"):
    """Save summary metrics to CSV."""
    import pandas as pd
    rows = []
    for name, r in zip(model_names, results_list):
        rows.append({
            "model":     name,
            "accuracy":  round(r["accuracy"],  4),
            "precision": round(r["precision"], 4),
            "recall":    round(r["recall"],    4),
            "f1":        round(r["f1"],        4),
            "itr":       round(r["itr"],       2),
        })
    df = pd.DataFrame(rows)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    log.info("Metrics saved to %s", filepath)
    return df
