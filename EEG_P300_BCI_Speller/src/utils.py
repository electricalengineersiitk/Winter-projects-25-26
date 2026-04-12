import os
import warnings
import numpy as np
import mne

def setup_environment():
    mne.set_log_level('WARNING')
    warnings.filterwarnings('ignore')
    os.makedirs('results', exist_ok=True)

def get_symbol_itr(n, acc, dur=2.1):
    if acc <= 1/n: return 0
    if acc == 1.0: acc = 0.99
    bits = np.log2(n) + acc * np.log2(acc) + (1 - acc) * np.log2((1 - acc) / (n - 1))
    return bits * (60.0 / dur)

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
