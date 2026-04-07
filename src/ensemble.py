import numpy as np
import mne
import warnings
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

# Local Imports (Zero-Leakage & Modular)
from preprocess import get_clean_data, apply_bad_channel_interpolation, apply_spatial_ica
from evaluate import get_character_prediction, get_symbol_itr

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

def run_ensemble_benchmark():
    """
    Final Ensemble Benchmark: 
    Uses 5-fold Stratified CV with full fold-specific preprocessing 
    (Bad Channels + ICA) to compute a scientifically valid Character ITR.
    """
    ds_name = "BNCI2014_009"
    subj = 1
    print(f"--- [ Ensemble ] {ds_name} Subject {subj} ---")
    
    epochs, X, y = get_clean_data(ds_name, subj)
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    
    # Model: Standard RBF-SVM Pipeline
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced'))
    ])
    
    subject_probs = []
    subject_y = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold+1}/5...")
        # Data Slicing
        e_tr, e_te = epochs[train_idx].copy(), epochs[test_idx].copy()
        y_tr, y_te = y[train_idx], y[test_idx]

        # 1. Zero-Leakage Preprocessing (Per-fold)
        e_tr, e_te = apply_bad_channel_interpolation(e_tr, e_te)
        e_tr, e_te = apply_spatial_ica(e_tr, e_te)
        
        # 2. Features
        X_tr = e_tr.get_data().reshape(len(e_tr), -1)
        X_te = e_te.get_data().reshape(len(e_te), -1)

        # 3. Classify
        clf.fit(X_tr, y_tr)
        subject_probs.extend(clf.predict_proba(X_te)[:, 1])
        subject_y.extend(y_te)

    # 4. Final Character-Level Analysis
    probs = np.array(subject_probs)
    y_test = np.array(subject_y)
    
    char_acc = get_character_prediction(probs, y_test, flash_per_char=12)
    itr_n36 = get_symbol_itr(36, char_acc, dur=12.0)

    print("\n--- FINAL ENSEMBLE REPORT ---")
    print(f"Character Accuracy (N=36): {char_acc*100:.1f}%")
    print(f"Communication Rate:        {itr_n36:.2f} bits/min")
    print(f"Protocol:                  5-Fold CV, per-fold ICA + BadChan")
    print("----------------------------\n")

if __name__ == "__main__":
    run_ensemble_benchmark()
