"""
train_and_evaluate.py
=====================
End-to-end training pipeline for the EEG Brain Speller.

Workflow
--------
1. Download BNCI2014-009 (P300 dataset) via MOABB
2. Load, preprocess, and epoch the data
3. Extract features (Xdawn + downsampling)
4. Train and cross-validate LDA, SVM, and EEGNet
5. Evaluate on held-out test subjects
6. Compute ITR and plot results

Usage
-----
    python train_and_evaluate.py [--model lda|svm|eegnet|all]
                                 [--subject 1..10]
                                 [--folds 5]
"""

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Local modules
from src.preprocess import preprocess_raw, make_epochs
from src.features   import XdawnFeatures, EpochDownsampleFeatures
from src.models     import build_lda, build_svm, get_model
from src.evaluate   import (
    cross_validate_model, evaluate_on_test,
    compute_itr, plot_confusion_matrix,
    plot_accuracy_vs_itr, save_metrics_csv, plot_erp
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── P300 speller constants ────────────────────────────────────────────────
N_SYMBOLS         = 36       # 6×6 character matrix
TRIAL_DURATION_S  = 2.0      # seconds per character decision
TEST_SIZE         = 0.2      # 20% held-out test
RANDOM_STATE      = 42


# ---------------------------------------------------------------------------
# Dataset loading via MOABB
# ---------------------------------------------------------------------------

def load_moabb_dataset(dataset_name: str = "BNCI2014_009",
                        subject_ids: list[int] | None = None):
    """
    Load a MOABB dataset and return raw epochs as numpy arrays.

    Returns
    -------
    X : (n_epochs, n_channels, n_times)  float32
    y : (n_epochs,)                      int
    meta : dict with dataset metadata
    """
    log.info("Loading MOABB dataset: %s", dataset_name)
    try:
        from moabb.datasets import BNCI2014_009
        from moabb.paradigms import P300
    except ImportError:
        raise ImportError("Install moabb:  pip install moabb")

    dataset   = BNCI2014_009()
    paradigm  = P300()

    if subject_ids is None:
        subject_ids = dataset.subject_list[:3]   # default: first 3 subjects

    log.info("Subjects: %s", subject_ids)
    X, y, meta = paradigm.get_data(dataset, subjects=subject_ids)

    # Convert string labels to integers
    classes = sorted(set(y))
    label_map = {c: i for i, c in enumerate(classes)}
    y_int = np.array([label_map[yi] for yi in y], dtype=int)

    log.info("Dataset loaded: X=%s  y=%s  classes=%s",
             X.shape, y_int.shape, classes)
    return X.astype(np.float32), y_int, {"classes": classes, "meta": meta}


# ---------------------------------------------------------------------------
# Feature extraction pipeline builders
# ---------------------------------------------------------------------------

def build_xdawn_lda_pipeline(n_components: int = 6,
                              target_samples: int = 30) -> Pipeline:
    return Pipeline([
        ("xdawn",   XdawnFeatures(n_components=n_components,
                                  target_samples=target_samples)),
        ("lda",     build_lda()),
    ])


def build_xdawn_svm_pipeline(n_components: int = 6,
                              target_samples: int = 30) -> Pipeline:
    return Pipeline([
        ("xdawn",   XdawnFeatures(n_components=n_components,
                                  target_samples=target_samples)),
        ("svm",     build_svm()),
    ])


def build_flat_lda_pipeline(target_samples: int = 30) -> Pipeline:
    return Pipeline([
        ("downsample", EpochDownsampleFeatures(target_samples=target_samples)),
        ("lda",        build_lda()),
    ])


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def run_experiment(model_names: list[str] = None,
                   subject_ids: list[int] | None = None,
                   k_folds: int = 5):
    """
    Full training and evaluation experiment.

    Parameters
    ----------
    model_names : list of model names to train/evaluate
    subject_ids : MOABB subject IDs to include
    k_folds     : number of cross-validation folds
    """
    if model_names is None:
        model_names = ["lda", "svm"]

    # ── 1. Load data ────────────────────────────────────────────────────
    X, y, meta = load_moabb_dataset(subject_ids=subject_ids)

    # ── 2. Train/test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE
    )
    log.info("Train: %d  |  Test: %d", len(y_train), len(y_test))

    # ── 3. Build model pipelines ─────────────────────────────────────────
    pipeline_map = {
        "lda":    build_xdawn_lda_pipeline(),
        "svm":    build_xdawn_svm_pipeline(),
        "lda_flat": build_flat_lda_pipeline(),
    }

    # EEGNet — raw 3-D input (no Xdawn flattening)
    try:
        from src.models import EEGNetClassifier
        n_ch, n_times = X.shape[1], X.shape[2]
        n_classes = len(set(y))
        pipeline_map["eegnet"] = EEGNetClassifier(
            n_classes=n_classes,
            n_channels=n_ch,
            n_times=n_times,
            n_epochs=50
        )
    except Exception as e:
        log.warning("EEGNet unavailable: %s", e)

    # Filter to requested models
    selected = {
        name: pipeline_map[name]
        for name in model_names
        if name in pipeline_map
    }

    all_results    = []
    all_cv_results = []

    for name, model in selected.items():
        log.info("=" * 55)
        log.info("Model: %s", name.upper())
        log.info("=" * 55)

        # EEGNet needs raw (n_epochs, n_ch, n_times);
        # sklearn pipelines need (n_epochs, features) — Xdawn handles this
        if name == "eegnet":
            Xtr, Xte = X_train, X_test
        else:
            Xtr, Xte = X_train, X_test

        # Cross-validation (on training set)
        log.info("Cross-validation on training set …")
        cv_res = cross_validate_model(model, Xtr, y_train, k=k_folds)
        all_cv_results.append(cv_res)

        # Final evaluation on held-out test set
        log.info("Evaluating on held-out test set …")
        result = evaluate_on_test(
            model, Xtr, y_train, Xte, y_test,
            n_symbols=N_SYMBOLS,
            trial_duration_s=TRIAL_DURATION_S
        )
        all_results.append(result)

        # Confusion matrix plot
        cm_path = RESULTS_DIR / f"confusion_matrix_{name}.png"
        plot_confusion_matrix(
            result["confusion_matrix"],
            class_names=[str(c) for c in meta["classes"]],
            title=f"Confusion Matrix — {name.upper()}",
            save_path=str(cm_path)
        )
        plt.close("all")

    # ── 4. Cross-model comparison plot ───────────────────────────────────
    names_used = list(selected.keys())
    comp_path  = RESULTS_DIR / "accuracy_vs_itr.png"
    plot_accuracy_vs_itr(all_results, names_used, save_path=str(comp_path))
    plt.close("all")

    # ── 5. Save metrics CSV ───────────────────────────────────────────────
    metrics_df = save_metrics_csv(
        all_results, names_used,
        filepath=str(RESULTS_DIR / "metrics.csv")
    )
    print("\n── Summary Metrics ──────────────────────────────────")
    print(metrics_df.to_string(index=False))
    print("─────────────────────────────────────────────────────\n")

    return all_results, metrics_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG Brain Speller — Training & Evaluation"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="all",
        help="Model(s) to train: lda | svm | eegnet | lda_flat | all"
    )
    parser.add_argument(
        "--subject", "-s", type=int, nargs="+", default=None,
        help="MOABB subject IDs (e.g. --subject 1 2 3)"
    )
    parser.add_argument(
        "--folds", "-k", type=int, default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    args = parser.parse_args()

    available_models = ["lda", "svm", "eegnet", "lda_flat"]
    models_to_run    = available_models if args.model == "all" \
                       else [args.model]

    run_experiment(
        model_names=models_to_run,
        subject_ids=args.subject,
        k_folds=args.folds
    )
