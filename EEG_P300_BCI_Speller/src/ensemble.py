import numpy as np
import mne
import warnings
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from preprocess import get_clean_data, apply_bad_channel_interpolation, apply_spatial_ica
from evaluate import get_symbol_itr
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

def get_character_prediction(probs, y_test, flash_ids, n_reps=10):
    unique_f = np.sort(np.unique(flash_ids))
    if len(unique_f) < 12: return 0 
    rows_ids = unique_f[:6]
    cols_ids = unique_f[6:]
    flash_per_char = 12 * n_reps
    n_chars = len(probs) // flash_per_char
    correct_chars = 0
    for i in range(n_chars):
        start = i * flash_per_char
        end = (i + 1) * flash_per_char
        char_probs = probs[start:end]
        char_labels = y_test[start:end]
        char_flashes = flash_ids[start:end]
        agg_probs = {}
        target_row = -1
        target_col = -1
        for p, l, f in zip(char_probs, char_labels, char_flashes):
            if f not in agg_probs: agg_probs[f] = []
            agg_probs[f].append(p)
            if l == 1:
                if f in rows_ids: target_row = f
                if f in cols_ids: target_col = f
        mean_probs = {f: np.mean(v) for f, v in agg_probs.items()}
        pred_row = rows_ids[np.argmax([mean_probs.get(r, 0) for r in rows_ids])]
        pred_col = cols_ids[np.argmax([mean_probs.get(c, 0) for c in cols_ids])]
        if pred_row == target_row and pred_col == target_col:
            correct_chars += 1
    return correct_chars / n_chars if n_chars > 0 else 0

def run_ensemble_benchmark():
    ds_name = "BNCI2014_009"
    subj = 1
    print(f"--- [ Ensemble ] {ds_name} Subject {subj} ---")
    epochs, X, y = get_clean_data(ds_name, subj)
    skf = StratifiedGroupKFold(n_splits=5)
    groups = epochs.metadata['char_id'].values
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced'))
    ])
    subject_probs = []
    subject_y = []
    subject_flashes = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y, groups=groups)):
        print(f"  Fold {fold+1}/5...")
        e_tr, e_te = epochs[train_idx].copy(), epochs[test_idx].copy()
        y_tr, y_te = y[train_idx], y[test_idx]
        e_tr, e_te = apply_bad_channel_interpolation(e_tr, e_te)
        e_tr.set_eeg_reference('average', verbose=False)
        e_te.set_eeg_reference('average', verbose=False)
        e_tr, e_te = apply_spatial_ica(e_tr, e_te)
        X_tr = e_tr.get_data().reshape(len(e_tr), -1)
        X_te = e_te.get_data().reshape(len(e_te), -1)
        clf.fit(X_tr, y_tr)
        subject_probs.extend(clf.predict_proba(X_te)[:, 1])
        subject_y.extend(y_te)
        subject_flashes.extend(e_te.metadata['flash_id'].values)
    probs = np.array(subject_probs)
    y_test = np.array(subject_y)
    flashes = np.array(subject_flashes)
    char_acc = get_character_prediction(probs, y_test, flashes)
    itr_n36 = get_symbol_itr(36, char_acc, dur=21.0)
    print("\n--- FINAL ENSEMBLE REPORT ---")
    print(f"Character Accuracy (N=36): {char_acc*100:.1f}%")
    print(f"Communication Rate:        {itr_n36:.2f} bits/min")
    print(f"Protocol:                  5-Fold GroupKFold (Zero-Leakage)")
    print("----------------------------\n")
if __name__ == "__main__":
    run_ensemble_benchmark()
