"""
features.py
===========
Feature extraction for P300, SSVEP, and Motor Imagery paradigms.

Supported methods
-----------------
P300
  - Epoch downsampling (vectorised waveform)
  - Xdawn spatial filtering

SSVEP
  - FFT amplitude at stimulus frequency + harmonics
  - Canonical Correlation Analysis (CCA)

Motor imagery
  - Common Spatial Patterns (CSP)

All paradigms
  - Riemannian covariance matrices (via pyriemann)
"""

import numpy as np
import mne
import logging
from sklearn.base import BaseEstimator, TransformerMixin

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# P300 Features
# ---------------------------------------------------------------------------

class EpochDownsampleFeatures(BaseEstimator, TransformerMixin):
    """
    Flatten each epoch into a 1-D feature vector by optional downsampling.

    Parameters
    ----------
    target_samples : int
        Number of time samples to keep per channel after decimation.
        Set to None to keep all samples.
    """

    def __init__(self, target_samples: int = 30):
        self.target_samples = target_samples

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : shape (n_epochs, n_channels, n_times)

        Returns
        -------
        features : shape (n_epochs, n_channels * target_samples)
        """
        n_epochs, n_ch, n_times = X.shape
        if self.target_samples and self.target_samples < n_times:
            indices = np.linspace(0, n_times - 1, self.target_samples, dtype=int)
            X = X[:, :, indices]
        return X.reshape(n_epochs, -1)


class XdawnFeatures(BaseEstimator, TransformerMixin):
    """
    Apply Xdawn spatial filtering before flattening.

    Uses MNE's Xdawn implementation which maximises the signal-to-noise
    ratio of the P300 ERP component.

    Parameters
    ----------
    n_components : int
        Number of Xdawn components to keep (default: 6).
    target_samples : int
        Downsampling after spatial filtering.
    """

    def __init__(self, n_components: int = 6, target_samples: int = 30):
        self.n_components = n_components
        self.target_samples = target_samples
        self._xdawn = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        from mne.preprocessing import Xdawn
        # MNE Xdawn expects MNE Epochs; we wrap the array manually.
        self._xdawn = Xdawn(n_components=self.n_components)
        # Create a minimal info for fitting
        info = mne.create_info(
            ch_names=[f"EEG{i:03d}" for i in range(X.shape[1])],
            sfreq=256, ch_types="eeg"
        )
        epochs_tmp = mne.EpochsArray(X, info, verbose=False)
        self._xdawn.fit(epochs_tmp)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        info = mne.create_info(
            ch_names=[f"EEG{i:03d}" for i in range(X.shape[1])],
            sfreq=256, ch_types="eeg"
        )
        epochs_tmp = mne.EpochsArray(X, info, verbose=False)
        epochs_xd = self._xdawn.transform(epochs_tmp)
        if hasattr(epochs_xd, 'get_data'):
            X_filt = epochs_xd.get_data()
        else:
            X_filt = epochs_xd
        n_epochs, n_ch, n_times = X_filt.shape
        if self.target_samples and self.target_samples < n_times:
            indices = np.linspace(0, n_times - 1, self.target_samples, dtype=int)
            X_filt = X_filt[:, :, indices]
        return X_filt.reshape(n_epochs, -1)


# ---------------------------------------------------------------------------
# SSVEP Features
# ---------------------------------------------------------------------------

class SSVEPFFTFeatures(BaseEstimator, TransformerMixin):
    """
    Extract FFT amplitude at stimulus frequency and its harmonics.

    Parameters
    ----------
    sfreq         : sampling frequency (Hz)
    stim_freqs    : list of stimulus frequencies in Hz, e.g. [8.0, 10.0, 12.0]
    n_harmonics   : number of harmonics (including fundamental)
    """

    def __init__(self, sfreq: float = 256.0,
                 stim_freqs: list[float] | None = None,
                 n_harmonics: int = 2):
        self.sfreq = sfreq
        self.stim_freqs = stim_freqs or [8.0, 10.0, 12.0, 15.0]
        self.n_harmonics = n_harmonics

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X : (n_epochs, n_channels, n_times)
        Returns : (n_epochs, n_channels * n_freqs * n_harmonics)
        """
        n_epochs, n_ch, n_times = X.shape
        freq_resolution = self.sfreq / n_times
        freqs = np.fft.rfftfreq(n_times, d=1.0 / self.sfreq)
        fft_vals = np.abs(np.fft.rfft(X, axis=-1))  # (n_ep, n_ch, n_freq_bins)

        features = []
        for epoch_fft in fft_vals:                   # (n_ch, n_freq_bins)
            row = []
            for sf in self.stim_freqs:
                for h in range(1, self.n_harmonics + 1):
                    target = sf * h
                    idx = int(round(target / freq_resolution))
                    idx = np.clip(idx, 0, epoch_fft.shape[-1] - 1)
                    row.append(epoch_fft[:, idx])    # (n_ch,)
            features.append(np.concatenate(row))
        return np.array(features)


class CCAFeatures(BaseEstimator, TransformerMixin):
    """
    Canonical Correlation Analysis features for SSVEP.

    For each stimulus frequency, compute CCA between multichannel EEG
    and a reference signal (sin+cos at fundamental + harmonics).
    The maximum canonical correlation becomes the feature for that frequency.

    Parameters
    ----------
    sfreq         : sampling frequency (Hz)
    stim_freqs    : stimulus frequencies
    n_harmonics   : number of harmonics for reference signals
    """

    def __init__(self, sfreq: float = 256.0,
                 stim_freqs: list[float] | None = None,
                 n_harmonics: int = 2):
        self.sfreq = sfreq
        self.stim_freqs = stim_freqs or [8.0, 10.0, 12.0, 15.0]
        self.n_harmonics = n_harmonics

    def _build_reference(self, freq: float, n_times: int) -> np.ndarray:
        """Build sin/cos reference matrix shape (2*n_harmonics, n_times)."""
        t = np.arange(n_times) / self.sfreq
        refs = []
        for h in range(1, self.n_harmonics + 1):
            refs.append(np.sin(2 * np.pi * freq * h * t))
            refs.append(np.cos(2 * np.pi * freq * h * t))
        return np.array(refs)                         # (2*n_harmonics, n_times)

    @staticmethod
    def _cca_corr(X: np.ndarray, Y: np.ndarray) -> float:
        """Return maximum canonical correlation between X and Y."""
        from sklearn.cross_decomposition import CCA
        n_comp = min(X.shape[0], Y.shape[0], 1)
        cca = CCA(n_components=n_comp)
        try:
            cca.fit(X.T, Y.T)
            X_c, Y_c = cca.transform(X.T, Y.T)
            return float(np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1])
        except Exception:
            return 0.0

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X : (n_epochs, n_channels, n_times)
        Returns: (n_epochs, n_stim_freqs)
        """
        n_epochs, n_ch, n_times = X.shape
        refs = {f: self._build_reference(f, n_times) for f in self.stim_freqs}
        features = np.zeros((n_epochs, len(self.stim_freqs)))
        for i, epoch in enumerate(X):
            for j, sf in enumerate(self.stim_freqs):
                features[i, j] = self._cca_corr(epoch, refs[sf])
        return features


# ---------------------------------------------------------------------------
# Motor Imagery Features — CSP
# ---------------------------------------------------------------------------

class CSPFeatures(BaseEstimator, TransformerMixin):
    """
    Common Spatial Patterns (CSP) for motor imagery.

    Wraps sklearn-compatible mne.decoding.CSP.

    Parameters
    ----------
    n_components : int  — number of CSP components (default: 6)
    log_var      : bool — return log-variance of filtered signals
    """

    def __init__(self, n_components: int = 6, log_var: bool = True):
        self.n_components = n_components
        self.log_var = log_var
        self._csp = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        from mne.decoding import CSP
        self._csp = CSP(n_components=self.n_components,
                        log=self.log_var,
                        reg=None)
        self._csp.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._csp.transform(X)


# ---------------------------------------------------------------------------
# Riemannian Covariance Features
# ---------------------------------------------------------------------------

class RiemannianFeatures(BaseEstimator, TransformerMixin):
    """
    Compute tangent-space features via Riemannian geometry.

    Requires pyriemann.

    Parameters
    ----------
    metric : str — covariance estimator (default: 'lwf' = Ledoit-Wolf)
    """

    def __init__(self, metric: str = "lwf"):
        self.metric = metric
        self._mapper = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        self._cov = Covariances(estimator=self.metric)
        self._ts = TangentSpace()
        covs = self._cov.fit_transform(X)
        self._ts.fit(covs)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        covs = self._cov.transform(X)
        return self._ts.transform(covs)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

FEATURE_REGISTRY = {
    "downsample": EpochDownsampleFeatures,
    "xdawn":      XdawnFeatures,
    "fft":        SSVEPFFTFeatures,
    "cca":        CCAFeatures,
    "csp":        CSPFeatures,
    "riemannian": RiemannianFeatures,
}


def get_feature_extractor(name: str, **kwargs):
    """Return an instantiated feature extractor by name."""
    if name not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown extractor '{name}'. "
                         f"Available: {list(FEATURE_REGISTRY)}")
    return FEATURE_REGISTRY[name](**kwargs)
