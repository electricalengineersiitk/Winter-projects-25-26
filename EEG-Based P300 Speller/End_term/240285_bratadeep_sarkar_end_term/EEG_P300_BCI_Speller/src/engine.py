import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import config
from preprocess import run_preprocessing_fold
from features import apply_xdawn, extract_riemannian_covariances, extract_p300_features

def run_model_evaluation(epochs, X, y, clf, name):
    """
    Standardized evaluation engine for all P300 classifiers.
    Handles per-fold cross-validation, preprocessing, and feature extraction.
    """
    groups = epochs.metadata['char_id'].values
    skf = StratifiedGroupKFold(n_splits=5)
    
    metrics = []
    probs = []
    true_y = []
    flash_ids = []
    
    for train_idx, test_idx in skf.split(X, y, groups=groups):
        e_tr = epochs[train_idx].copy()
        e_te = epochs[test_idx].copy()
        y_tr = y[train_idx]
        y_te = y[test_idx]
        
        # Consistent per-fold preprocessing (AutoReject)
        e_tr, e_te = run_preprocessing_fold(e_tr, e_te)
        
        # Model-specific feature extraction
        if "Xdawn" in name:
            X_tr, X_te = apply_xdawn(e_tr, y_tr, e_te, decimation_factor=config.DECIMATION_FACTOR)
            
        elif "Riemannian" in name:
            X_tr = extract_riemannian_covariances(e_tr.get_data())
            X_te = extract_riemannian_covariances(e_te.get_data())
            
        elif "EEGNet" in name:
            raw_tr = e_tr.get_data()
            raw_te = e_te.get_data()
            mu, sd = np.mean(raw_tr), np.std(raw_tr)
            X_tr = ((raw_tr - mu) / sd)[:, np.newaxis, :, :].astype(np.float32)
            X_te = ((raw_te - mu) / sd)[:, np.newaxis, :, :].astype(np.float32)
            y_tr = y_tr.astype(np.int64)
            y_te = y_te.astype(np.int64)
            
        else:
            # LDA / SVM
            X_tr = extract_p300_features(e_tr.get_data(), decimation_factor=config.DECIMATION_FACTOR)
            X_te = extract_p300_features(e_te.get_data(), decimation_factor=config.DECIMATION_FACTOR)
            
        # Training
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        
        # Robust Probability/Score extraction
        try:
            # Standard classifiers (LDA, SVM, EEGNet)
            fold_probs = clf.predict_proba(X_te)[:, 1]
        except (AttributeError, NotImplementedError):
            try:
                # Riemannian MDM / SVM with no proba
                scores = clf.decision_function(X_te)
                fold_probs = 1.0 / (1.0 + np.exp(-scores))  # sigmoid normalization
            except (AttributeError, NotImplementedError):
                # Hard fallback
                fold_probs = y_pred.astype(float)
        
        probs.extend(fold_probs)
        true_y.extend(y_te)
        flash_ids.extend(e_te.metadata['flash_id'].values)
        
        metrics.append([
            accuracy_score(y_te, y_pred),
            recall_score(y_te, y_pred, average='binary', zero_division=0),
            precision_score(y_te, y_pred, average='binary', zero_division=0),
            f1_score(y_te, y_pred, average='binary', zero_division=0),
        ])
        
    return {
        'metrics': np.mean(metrics, axis=0),
        'probs': np.array(probs),
        'true_y': np.array(true_y),
        'flash_ids': np.array(flash_ids)
    }
