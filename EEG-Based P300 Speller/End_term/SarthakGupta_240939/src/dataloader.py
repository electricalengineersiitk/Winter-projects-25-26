import os
import scipy.io as sio
import numpy as np

def load_file(path):
    data=sio.loadmat(path)
    signal=data['Signal'].astype(np.float64)
    stimulus=data['StimulusCode'].astype(int)
    flashing=data['Flashing'].astype(int)
    stim_type=data['StimulusType'].astype(int) if 'StimulusType' in data else None
    target=data['TargetChar'][0] if 'TargetChar' in data else None
    return {
        "signal":signal,
        "stimulus":stimulus,
        "flashing":flashing,
        "stim_type":stim_type,
        "target":target,
        "file":os.path.basename(path)
    }

def load_all(data_dir="data"):
    files=[
        "Subject_A_Train.mat",
        "Subject_A_Test.mat",
        "Subject_B_Train.mat",
        "Subject_B_Test.mat"
    ]
    datasets={}
    for f in files:
        path=os.path.join(data_dir,f)
        if not os.path.exists(path):
            print(f"Skipping missing file: {f}")
            continue
        print(f"Loading: {f}")
        datasets[f]=load_file(path)
    return datasets