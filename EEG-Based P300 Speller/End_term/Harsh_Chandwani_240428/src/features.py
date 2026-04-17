import numpy as np
from pyriemann.spatialfilters import Xdawn

xd = None

def transform_epochs(all_segments, y_all=None, fit=False, mode="lda"):
    global xd
    
    if mode == "eegnet":
        X_per_char = [np.array([seg[20:180:2, :].T for seg in segs]) for segs in all_segments]
        if fit:
            return np.concatenate(X_per_char), X_per_char
        return X_per_char
    
    X_flat = np.array([seg[20:120:4, :].T for segs in all_segments for seg in segs])
    
    if fit:
        xd = Xdawn(nfilter=6)
        xd.fit(X_flat, y_all)
        
    X_per_char, idx = [], 0
    
    for segs in all_segments:
        n = len(segs)
        X_char = xd.transform(X_flat[idx:idx+n])
        X_per_char.append(X_char.reshape(X_char.shape[0], -1))
        idx += n
    
    if fit:
        return np.concatenate(X_per_char), X_per_char
    return X_per_char