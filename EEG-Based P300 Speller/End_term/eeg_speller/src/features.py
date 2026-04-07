
"""
features.py — Feature extraction for P300 EEG epochs
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Downsample + flatten 
def downsample_flatten(X: np.ndarray, target_samples: int = 30) -> np.ndarray:
    """
    Subsample along the time axis then flatten channels × time.
    Input : (n_epochs, n_channels, n_times)
    Output: (n_epochs, n_channels * target_samples)
    """
    n_epochs, n_channels, n_times = X.shape
    idx        = np.linspace(0, n_times - 1, target_samples, dtype=int)
    subsampled = X[:, :, idx] # (n, ch, target_samples)
    return subsampled.reshape(n_epochs, -1)

# Xdawn spatial filter (pyriemann)
def xdawn_features(X: np.ndarray, y: np.ndarray,
                   n_filters: int = 8) -> tuple:
    """
    Apply Xdawn spatial filtering to enhance P300 SNR.
    Falls back to PCA if pyriemann is not installed.
    Returns (X_feat, transformer)
    """
    try:
        from pyriemann.estimation import Xdawn
        xd  = Xdawn(nfilter=n_filters)
        out = xd.fit_transform(X, y)   # (n, n_filters, n_times)
        return out.reshape(len(X), -1), xd
    except ImportError:
        print("pyriemann not found — falling back to PCA.")
        return pca_features(X, n_components=n_filters * X.shape[2])

# PCA 
def pca_features(X: np.ndarray,
                 n_components: int = 50) -> tuple:
    """
    Flatten then reduce with PCA.
    Returns (X_feat, pca_object)
    """
    flat = X.reshape(len(X), -1)
    n    = min(n_components, flat.shape[1])
    pca  = PCA(n_components=n)
    return pca.fit_transform(flat), pca

# Unified entry point 
def extract_features(X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_test:  np.ndarray,
                     method:  str  = 'downsample',
                     target_samples: int = 30,
                     n_components:   int = 8) -> tuple:
    """
    Extract + scale features for train and test sets.
    method : 'downsample' | 'xdawn' | 'pca'
    Returns
    X_tr_feat, X_te_feat, scaler, transformer
    """
    if method == 'xdawn':
        X_tr_feat, tf = xdawn_features(X_train, y_train, n_filters=n_components)
        X_te_feat      = tf.transform(X_test).reshape(len(X_test), -1)

    elif method == 'pca':
        flat_tr        = X_train.reshape(len(X_train), -1)
        flat_te        = X_test.reshape(len(X_test),  -1)
        pca            = PCA(n_components=min(n_components, flat_tr.shape[1]))
        X_tr_feat      = pca.fit_transform(flat_tr)
        X_te_feat      = pca.transform(flat_te)
        tf             = pca

    else:   # default: downsample
        X_tr_feat = downsample_flatten(X_train, target_samples)
        X_te_feat = downsample_flatten(X_test,  target_samples)
        tf        = None

    # Standard scaling
    scaler    = StandardScaler()
    X_tr_feat = scaler.fit_transform(X_tr_feat)
    X_te_feat = scaler.transform(X_te_feat)

    return X_tr_feat, X_te_feat, scaler, tf