import numpy as np
from scipy.io import loadmat

def load_subject(path):
    data = loadmat(path)

    signal    = data['Signal'].astype(np.float64)
    flashing  = data['Flashing'].astype(int)
    stim_code = data['StimulusCode'].astype(int)
    stim_type = data['StimulusType'].astype(int) if 'StimulusType' in data else None
    target_char = data['TargetChar'][0] if 'TargetChar' in data else None

    print(f"\nLoaded: {path}")
    print("Signal:", signal.shape)

    return signal, flashing, stim_code, stim_type, target_char