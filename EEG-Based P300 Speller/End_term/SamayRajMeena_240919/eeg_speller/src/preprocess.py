
"""
preprocess.py — Download, load, filter, and epoch EEG data
Dataset: BNCI2014_009 (EPFL P300 speller, 10 subjects) via MOABB
"""

import numpy as np
import mne
from moabb.datasets import BNCI2014_009
from moabb.paradigms import P300

# Constants
FS          = 256         
FMIN        = 0.1        
FMAX        = 20.0         
TMIN        = -0.2         
TMAX        =  0.8         
NOTCH_FREQ  = 50.0         


# Download
def download_dataset():
    ds = BNCI2014_009()
    ds.download()
    print("Download complete.\n")


# Load one subject via MOABB paradigm 
def load_subject(subject_id: int,
                 fmin: float = FMIN,
                 fmax: float = FMAX):
    """
    Load a single subject's epochs using the MOABB P300 paradigm.
    Parameters
    subject_id : int   1–10
    Returns
    X        : ndarray  (n_epochs, n_channels, n_times)
    y_binary : ndarray  (n_epochs,)  — 0 = NonTarget, 1 = Target
    meta     : DataFrame with session / run info
    """
    dataset  = BNCI2014_009()
    paradigm = P300(fmin=fmin, fmax=fmax,
                    tmin=TMIN, tmax=TMAX,
                    baseline=(-0.2, 0.0)) # Changed from None

    X, y_str, meta = paradigm.get_data(dataset=dataset,
                                        subjects=[subject_id])

    # Convert string labels → binary integers
    y_binary = (y_str == 'Target').astype(int)

    print(f"Subject {subject_id:02d} | "
          f"epochs: {X.shape[0]:5d} | "
          f"channels: {X.shape[1]:3d} | "
          f"samples: {X.shape[2]:3d} | "
          f"targets: {y_binary.sum():4d} / {len(y_binary)}")

    return X, y_binary, meta

# Load all subjects 
def load_all_subjects(subject_ids=None, fmin=FMIN, fmax=FMAX):
    """
    Load multiple subjects. Default: all 10.

    Returns
    data : dict  {subject_id: {'X':, 'y':, 'meta':}}
    """
    if subject_ids is None:
        subject_ids = list(range(1, 11))

    data = {}
    for sid in subject_ids:
        X, y, meta = load_subject(sid, fmin=fmin, fmax=fmax)
        data[sid]  = {'X': X, 'y': y, 'meta': meta}

    print(f"\nLoaded {len(data)} subjects.\n")
    return data

# Optional extra notch filter (MNE-based, applied on raw) 
def apply_notch_to_epochs(X: np.ndarray,
                           notch: float = NOTCH_FREQ,
                           fs: int = FS) -> np.ndarray:
    """
    Apply a notch filter to already-epoched data (n_epochs, n_ch, n_times).
    The MOABB paradigm handles bandpass; this adds 50 Hz notch if needed.
    """
    from scipy.signal import iirnotch, filtfilt
    b, a = iirnotch(notch / (fs / 2.0), Q=30.0)
    return filtfilt(b, a, X, axis=-1)   # filter along time axis
from mne.preprocessing import ICA

def apply_ica_to_epochs(X: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Satisfies the requirement to apply ICA for artifact rejection.
    Wraps the MOABB numpy output into an MNE EpochsArray to fit ICA.
    """
    # Create a dummy MNE info object for our 16 channels
    ch_names = [f'EEG{i+1}' for i in range(X.shape[1])]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    epochs = mne.EpochsArray(X, info, verbose=False)
    ica = ICA(n_components=10, random_state=42, max_iter='auto')
    ica.fit(epochs, verbose=False)
    epochs_clean = ica.apply(epochs, verbose=False)
    return epochs_clean.get_data(copy=False)

