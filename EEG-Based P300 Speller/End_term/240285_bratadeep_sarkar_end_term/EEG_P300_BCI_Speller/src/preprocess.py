import mne
import numpy as np
import pandas as pd
from moabb.datasets import BNCI2014_009, EPFLP300
from mne.preprocessing import ICA
from autoreject import AutoReject
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

SEED = 42


def get_clean_data(dataset_name='BNCI2014_009', subj=1):
    """
    Load and preprocess continuous raw EEG signal, THEN epoch.
    Stage 1 (Raw): Bandpass, Notch, Re-reference, Bad Channel Detection, ICA.
    Stage 2 (Epoched): Epoch extraction with baseline correction.
    """
    if dataset_name == 'BNCI2014_009':
        ds = BNCI2014_009()
    elif dataset_name == 'EPFLP300':
        ds = EPFLP300()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    data = ds.get_data(subjects=[subj])[subj]
    s_key = list(data.keys())[0]
    r_key = list(data[s_key].keys())[0]
    raw = data[s_key][r_key]
    raw.pick_types(eeg=True)

    # --- Stage 1: Raw-Level Preprocessing ---

    # 1a. Bandpass filter (P300 band)
    raw.filter(0.1, 30.0, verbose=False)

    # 1b. Notch filter (Indian power-line noise at 50 Hz)
    raw.notch_filter(freqs=50, verbose=False)

    # 1c. Re-reference to average
    raw.set_eeg_reference('average', projection=True, verbose=False)

    # 1d. Bad Channel Detection on raw (z-score of variance across time)
    raw_data = raw.get_data()
    ch_vars = np.var(raw_data, axis=1)
    z = np.abs((ch_vars - np.median(ch_vars)) / (np.median(np.abs(ch_vars - np.median(ch_vars))) + 1e-8))
    bad_chs = [raw.ch_names[i] for i in np.where(z > 3.5)[0]]
    if bad_chs and len(bad_chs) < len(raw.ch_names) // 2:
        print(f"    -> Bad channels detected on raw: {bad_chs}")
        raw.info['bads'] = bad_chs
        raw.interpolate_bads(reset_bads=True, verbose=False)

    # 1e. ICA on continuous raw (removes eye-blinks/muscle globally before epoching)
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    n_comp = min(len(raw.ch_names) - 1, 15)
    ica = ICA(n_components=n_comp, random_state=SEED, method='fastica', max_iter=500)
    ica.fit(raw_for_ica, verbose=False)
    frontal_chs = [c for c in ['Fp1', 'Fp2', 'AF3', 'AF4', 'Fpz', 'FP1', 'FP2', 'FPZ']
                   if c in raw.ch_names]
    if frontal_chs:
        eog_idx, _ = ica.find_bads_eog(raw, ch_name=frontal_chs, verbose=False)
        ica.exclude = eog_idx
        print(f"    -> ICA (raw): Excluded {len(eog_idx)} blink components via {frontal_chs}")
    else:
        ica.exclude = []
        print(f"    -> ICA (raw): No frontal channels found; no components excluded.")
    ica.apply(raw, verbose=False)

    # --- Stage 2: Epoching ---
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    target_id = event_id.get('Target')
    nontarget_id = event_id.get('NonTarget')
    epochs = mne.Epochs(
        raw, events,
        event_id={'Target': target_id, 'NonTarget': nontarget_id},
        tmin=-0.2, tmax=0.8, baseline=(-0.2, 0),
        preload=True, verbose=False
    )

    # Build flash/char metadata for ensemble logic
    flash_ids = [i % 12 for i in range(len(events))]
    char_ids = np.arange(len(events)) // 12

    epochs.metadata = pd.DataFrame({
        'flash_id': flash_ids[:len(epochs)],
        'char_id':  char_ids[:len(epochs)]
    })

    y = (epochs.events[:, -1] == target_id).astype(int)
    return epochs, epochs.get_data(), y


def run_preprocessing_fold(epochs_train, epochs_test):
    """
    Per-fold preprocessing ONLY for AutoReject-based epoch cleaning.
    ICA and Bad Channel handling have already been applied globally in get_clean_data.
    """
    ar = AutoReject(random_state=SEED, n_jobs=1, verbose=False)
    ar.fit(epochs_train)
    epochs_train = ar.transform(epochs_train)
    epochs_test = ar.transform(epochs_test)
    return epochs_train, epochs_test
