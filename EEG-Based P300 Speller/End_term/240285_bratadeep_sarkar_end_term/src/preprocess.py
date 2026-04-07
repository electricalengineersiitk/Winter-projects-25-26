import mne
import numpy as np
from moabb.datasets import BNCI2014_009, EPFLP300
from mne.preprocessing import ICA
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
SEED = 42
def get_clean_data(dataset_name='BNCI2014_009', subj=1, apply_decimation=True):
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
    raw.filter(0.1, 30.0, verbose=False)
    raw.notch_filter(freqs=50, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    target_id = event_id.get('Target')
    nontarget_id = event_id.get('NonTarget')
    epochs = mne.Epochs(raw, events, event_id={'Target': target_id, 'NonTarget': nontarget_id},
                        tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), preload=True, verbose=False)
    stim_ch = next((raw.ch_names.index(c) for c in ['Flash stim', 'STI', 'stim'] if c in raw.ch_names), None)
    flash_ids = []
    if stim_ch is not None:
        for event_time in events[:, 0]:
            val = raw[stim_ch, event_time][0][0][0]
            flash_ids.append(int(val))
    else:
        flash_ids = [i % 12 for i in range(len(events))]
    char_ids = np.arange(len(events)) // 120
    epochs.metadata = mne.utils._prepare_metadata(
        metadata=np.column_stack([flash_ids, char_ids]),
        names=['flash_id', 'char_id'],
        col_type={'flash_id': 'int64', 'char_id': 'int64'},
        row_names=None
    )
    return epochs, epochs.get_data(), (epochs.events[:, -1] == target_id).astype(int)
def apply_bad_channel_interpolation(epochs_train, epochs_test, z_thresh=3.0):
    train_data = epochs_train.get_data() 
    chan_stds = np.std(train_data, axis=(0, 2))
    median_std = np.median(chan_stds)
    mad = np.median(np.abs(chan_stds - median_std))
    z_scores = np.abs(chan_stds - median_std) / (mad + 1e-8)
    bad_idx = np.where(z_scores > z_thresh)[0]
    bads = [epochs_train.ch_names[i] for i in bad_idx]
    if bads:
        if len(bads) < len(epochs_train.ch_names) // 2:
            epochs_train.info['bads'] = bads
            epochs_test.info['bads'] = bads
            epochs_train.interpolate_bads(reset_bads=True, verbose=False)
            epochs_test.interpolate_bads(reset_bads=True, verbose=False)
    return epochs_train, epochs_test
def apply_spatial_ica(epochs_train, epochs_test):
    ica = ICA(n_components=min(len(epochs_train.ch_names), 15), random_state=SEED, method='fastica')
    epochs_for_ica = epochs_train.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    ica.fit(epochs_for_ica, verbose=False)
    frontal_chans = [ch for ch in ['Fp1', 'Fp2', 'AF3', 'AF4', 'Fpz', 'FP1', 'FP2', 'FPZ'] if ch in epochs_train.ch_names]
    if frontal_chans:
        eog_indices, eog_scores = ica.find_bads_eog(epochs_train, ch_name=frontal_chans, verbose=False)
        ica.exclude = eog_indices
        if not ica.exclude:
            ica.exclude = []
        print(f"    -> ICA: Spatial Audit found {len(ica.exclude)} artifact components correlating with {frontal_chans}")
    else:
        ica.exclude = [] 
        print(f"    -> ICA: No frontal channels found. Excluding nothing.")
    ica.apply(epochs_train, verbose=False)
    ica.apply(epochs_test, verbose=False)
    return epochs_train, epochs_test
