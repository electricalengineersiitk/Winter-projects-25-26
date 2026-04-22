"""
models.py
=========
Classifiers for EEG BCI:
  - LDA (baseline)
  - SVM with RBF kernel
  - EEGNet (compact CNN, PyTorch)
  - Ensemble score averaging across flash repetitions

All sklearn-compatible estimators.
"""

import numpy as np
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------

def build_lda(solver: str = "svd", shrinkage=None) -> Pipeline:
    """
    LDA classifier wrapped in a standard scaler pipeline.

    Parameters
    ----------
    solver    : 'svd', 'lsqr', or 'eigen'
    shrinkage : None | 'auto' | float  (ignored for solver='svd')
    """
    lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lda",    lda),
    ])


def build_svm(C: float = 1.0,
              kernel: str = "rbf",
              gamma: str = "scale",
              probability: bool = True) -> Pipeline:
    """
    SVM classifier wrapped in a standard scaler pipeline.

    Parameters
    ----------
    C           : regularisation parameter
    kernel      : 'rbf', 'linear', or 'poly'
    gamma       : 'scale' or 'auto' or float
    probability : enable probability estimates (needed for score ensembling)
    """
    svm = SVC(C=C, kernel=kernel, gamma=gamma,
              probability=probability, class_weight="balanced")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    svm),
    ])


# ---------------------------------------------------------------------------
# EEGNet — PyTorch implementation
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    log.warning("PyTorch not available. EEGNet will not be usable.")


if _TORCH_AVAILABLE:

    class _EEGNetModule(nn.Module):
        """
        EEGNet: A Compact Convolutional Neural Network for EEG-Based BCIs.
        Lawhern et al. (2018) — https://arxiv.org/abs/1611.08024

        Architecture
        ------------
        Block 1: Temporal convolution → Depthwise spatial convolution → BN → ELU → AvgPool → Dropout
        Block 2: Separable depthwise convolution → BN → ELU → AvgPool → Dropout
        Classifier: Flatten → Dense → Softmax

        Parameters
        ----------
        n_classes   : number of output classes
        n_channels  : number of EEG channels
        n_times     : number of time samples per epoch
        F1          : number of temporal filters (default: 8)
        D           : depth multiplier for spatial filters (default: 2)
        F2          : number of separable filters (default: F1*D = 16)
        dropout_rate: dropout probability
        """

        def __init__(self,
                     n_classes: int = 2,
                     n_channels: int = 64,
                     n_times: int = 256,
                     F1: int = 8,
                     D: int = 2,
                     F2: int = 16,
                     dropout_rate: float = 0.5):
            super().__init__()
            F2 = F2 or F1 * D

            # Block 1 — Temporal convolution
            self.block1_conv = nn.Conv2d(
                1, F1, kernel_size=(1, n_times // 2),
                padding=(0, n_times // 4), bias=False
            )
            self.block1_bn1 = nn.BatchNorm2d(F1)
            # Depthwise spatial convolution
            self.block1_dw = nn.Conv2d(
                F1, F1 * D, kernel_size=(n_channels, 1),
                groups=F1, bias=False
            )
            self.block1_bn2 = nn.BatchNorm2d(F1 * D)
            self.block1_pool = nn.AvgPool2d(kernel_size=(1, 4))
            self.block1_drop = nn.Dropout(dropout_rate)

            # Block 2 — Separable convolution
            self.block2_dw = nn.Conv2d(
                F1 * D, F1 * D,
                kernel_size=(1, 16), padding=(0, 8),
                groups=F1 * D, bias=False
            )
            self.block2_pw = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
            self.block2_bn = nn.BatchNorm2d(F2)
            self.block2_pool = nn.AvgPool2d(kernel_size=(1, 8))
            self.block2_drop = nn.Dropout(dropout_rate)

            # Classifier head
            # Compute flattened feature size dynamically
            self._flat_size = self._get_flat_size(n_channels, n_times, F1, D, F2)
            self.fc = nn.Linear(self._flat_size, n_classes)

        def _get_flat_size(self, n_channels, n_times, F1, D, F2):
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self._forward_features(x)
            return int(np.prod(x.shape[1:]))

        def _forward_features(self, x):
            # Block 1
            x = self.block1_conv(x)
            x = self.block1_bn1(x)
            x = self.block1_dw(x)
            x = self.block1_bn2(x)
            x = F.elu(x)
            x = self.block1_pool(x)
            x = self.block1_drop(x)
            # Block 2
            x = self.block2_dw(x)
            x = self.block2_pw(x)
            x = self.block2_bn(x)
            x = F.elu(x)
            x = self.block2_pool(x)
            x = self.block2_drop(x)
            return x

        def forward(self, x):
            # x: (batch, n_channels, n_times)
            x = x.unsqueeze(1)              # → (batch, 1, n_channels, n_times)
            x = self._forward_features(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)               # logits

    class EEGNetClassifier(BaseEstimator, ClassifierMixin):
        """
        Sklearn-compatible wrapper around the PyTorch EEGNet module.

        Parameters
        ----------
        n_classes    : number of target classes
        n_channels   : EEG channel count
        n_times      : time samples per epoch
        F1, D, F2    : EEGNet architecture params
        dropout_rate : dropout probability
        lr           : learning rate
        n_epochs     : training epochs
        batch_size   : mini-batch size
        device       : 'cpu' | 'cuda' | 'mps'
        """

        def __init__(self,
                     n_classes: int = 2,
                     n_channels: int = 64,
                     n_times: int = 256,
                     F1: int = 8,
                     D: int = 2,
                     F2: int = 16,
                     dropout_rate: float = 0.5,
                     lr: float = 1e-3,
                     n_epochs: int = 100,
                     batch_size: int = 32,
                     device: str | None = None):
            self.n_classes = n_classes
            self.n_channels = n_channels
            self.n_times = n_times
            self.F1 = F1
            self.D = D
            self.F2 = F2
            self.dropout_rate = dropout_rate
            self.lr = lr
            self.n_epochs = n_epochs
            self.batch_size = batch_size
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model_ = None
            self.classes_ = None

        def _build_model(self):
            return _EEGNetModule(
                n_classes=self.n_classes,
                n_channels=self.n_channels,
                n_times=self.n_times,
                F1=self.F1, D=self.D, F2=self.F2,
                dropout_rate=self.dropout_rate,
            ).to(self.device)

        def fit(self, X: np.ndarray, y: np.ndarray):
            """
            X : (n_samples, n_channels, n_times)
            y : (n_samples,) integer class labels
            """
            self.classes_ = np.unique(y)
            self.model_ = self._build_model()
            optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
            criterion = nn.CrossEntropyLoss()

            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.long)

            dataset = torch.utils.data.TensorDataset(X_t, y_t)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )

            self.model_.train()
            for epoch in range(self.n_epochs):
                total_loss = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    logits = self.model_(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if (epoch + 1) % 10 == 0:
                    log.info("EEGNet epoch %d/%d — loss: %.4f",
                             epoch + 1, self.n_epochs, total_loss / len(loader))
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            self.model_.eval()
            with torch.no_grad():
                X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
                logits = self.model_(X_t)
                preds = logits.argmax(dim=1).cpu().numpy()
            return preds

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            self.model_.eval()
            with torch.no_grad():
                X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
                logits = self.model_(X_t)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            return probs

        def score(self, X: np.ndarray, y: np.ndarray) -> float:
            preds = self.predict(X)
            return float(np.mean(preds == y))


# ---------------------------------------------------------------------------
# Ensemble: average predicted probabilities across repetitions
# ---------------------------------------------------------------------------

class RepetitionEnsemble(BaseEstimator, ClassifierMixin):
    """
    Improve P300 speller accuracy by averaging classifier scores over
    multiple flash repetitions before making a character decision.

    The P300 speller flashes each row/column multiple times; averaging
    the soft scores across repetitions significantly boosts accuracy.

    Parameters
    ----------
    base_clf : fitted sklearn classifier with predict_proba()
    n_reps   : number of repetitions per character trial
    """

    def __init__(self, base_clf, n_reps: int = 10):
        self.base_clf = base_clf
        self.n_reps = n_reps

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.base_clf.fit(X, y)
        self.classes_ = self.base_clf.classes_
        return self

    def predict_proba_averaged(self, X: np.ndarray) -> np.ndarray:
        """
        Average probabilities over blocks of n_reps consecutive epochs.

        X : (n_epochs, n_features) or (n_epochs, n_ch, n_times)
        Returns averaged probabilities per block.
        """
        probs = self.base_clf.predict_proba(X)      # (n_epochs, n_classes)
        n_blocks = len(probs) // self.n_reps
        averaged = np.array([
            probs[i * self.n_reps:(i + 1) * self.n_reps].mean(axis=0)
            for i in range(n_blocks)
        ])
        return averaged

    def predict(self, X: np.ndarray) -> np.ndarray:
        avg = self.predict_proba_averaged(X)
        return self.classes_[avg.argmax(axis=1)]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "lda": build_lda,
    "svm": build_svm,
}

if _TORCH_AVAILABLE:
    MODEL_REGISTRY["eegnet"] = EEGNetClassifier


def get_model(name: str, **kwargs):
    """Return an instantiated model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
