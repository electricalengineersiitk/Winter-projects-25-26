import warnings
import numpy as np
import mne


def setup_environment():
    """Configure MNE logging, suppress warnings, and ensure results dir exists."""
    from config import RESULTS_DIR
    mne.set_log_level('WARNING')
    warnings.filterwarnings('ignore')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_symbol_itr(n, acc, dur=2.1):
    """
    Compute Information Transfer Rate (ITR) in bits/min.
    n   : number of symbols (N=36 for 6x6 P300 matrix)
    acc : character-level accuracy (0–1)
    dur : trial duration in seconds
    """
    if acc <= 1.0 / n:
        return 0.0
    if acc >= 1.0:
        acc = 0.9999
    bits = (np.log2(n)
            + acc * np.log2(acc)
            + (1 - acc) * np.log2((1 - acc) / (n - 1)))
    return bits * (60.0 / dur)


def get_character_prediction(probs, y_test, flash_ids, n_reps=None):
    """
    Decode character identity from accumulated flash probabilities.

    The P300 speller presents 6 rows + 6 columns = 12 flash groups per character.
    n_reps repetitions of those 12 groups are averaged per character.

    If n_reps is None, it is inferred automatically from the data by finding
    the number of flashes per unique character boundary. This handles datasets
    that differ in repetition count (e.g. BNCI=10, EPFLP300=15).
    """
    unique_f = np.sort(np.unique(flash_ids))
    if len(unique_f) < 12:
        return 0.0

    rows_ids = unique_f[:6]
    cols_ids = unique_f[6:]

    # --- Auto-detect n_reps if not provided ---
    if n_reps is None:
        # Count how many times flash_id 0 appears; that equals n_reps
        n_reps = int(np.sum(flash_ids == unique_f[0]))
        if n_reps == 0:
            n_reps = 10  # safe fallback

    flash_per_char = 12 * n_reps
    n_chars = len(probs) // flash_per_char
    if n_chars == 0:
        return 0.0

    correct_chars = 0
    for i in range(n_chars):
        start = i * flash_per_char
        end   = (i + 1) * flash_per_char

        char_probs  = probs[start:end]
        char_labels = y_test[start:end]
        char_flashes = flash_ids[start:end]

        agg_probs  = {}
        target_row = -1
        target_col = -1

        for p, l, f in zip(char_probs, char_labels, char_flashes):
            agg_probs.setdefault(f, []).append(p)
            if l == 1:
                if f in rows_ids:
                    target_row = f
                if f in cols_ids:
                    target_col = f

        mean_probs = {f: np.mean(v) for f, v in agg_probs.items()}
        pred_row = rows_ids[np.argmax([mean_probs.get(r, 0.0) for r in rows_ids])]
        pred_col = cols_ids[np.argmax([mean_probs.get(c, 0.0) for c in cols_ids])]

        if pred_row == target_row and pred_col == target_col:
            correct_chars += 1

    return correct_chars / n_chars
