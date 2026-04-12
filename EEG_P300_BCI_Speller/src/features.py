import numpy as np
from mne.preprocessing import Xdawn
try:
    from pyriemann.estimation import Covariances
except ImportError:
    Covariances = None

def extract_p300_features(epochs_data, decimation_factor=1):
    """
    Downsample epoch waveform into a flattened feature vector.
    """
    if decimation_factor > 1:
        return epochs_data[:, :, ::decimation_factor].reshape(len(epochs_data), -1)
    return epochs_data.reshape(len(epochs_data), -1)

def apply_xdawn(epochs_train, y_train, epochs_test, n_components=2):
    """
    Applies Xdawn spatial filtering to enhance the SNR of ERP components.
    Fits on train, transforms train and test.
    """
    xd = Xdawn(n_components=n_components, correct_overlap=False, reg=0.1)
    xd.fit(epochs_train, y_train)
    
    # Transform and flatten
    X_train_xd = xd.transform(epochs_train).reshape(len(epochs_train), -1)
    X_test_xd = xd.transform(epochs_test).reshape(len(epochs_test), -1)
    
    return X_train_xd, X_test_xd

def extract_riemannian_covariances(epochs_data, estimator='oas'):
    """
    Extracts spatial covariance matrices for Riemannian geometry classification.
    """
    if Covariances is None:
        raise ImportError("pyriemann is not installed. Please install it to use covariance features.")
    
    cov_estimator = Covariances(estimator=estimator)
    return cov_estimator.transform(epochs_data)
