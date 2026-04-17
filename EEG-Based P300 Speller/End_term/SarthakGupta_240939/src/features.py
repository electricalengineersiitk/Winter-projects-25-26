import numpy as np
from pyriemann.spatialfilters import Xdawn

xdawn=None

def extract_features(all_segments,y_all=None,fit=False,mode="lda"):
    global xdawn
    if mode=="eegnet":
        X_per_char=[]
        for segs in all_segments:
            X_char=np.array([seg[20:180:2,:].T for seg in segs])
            X_per_char.append(X_char)
        if fit:
            X_flat=np.concatenate(X_per_char)
            return X_flat,X_per_char
        return X_per_char
    def window(seg):
        return seg[20:120:4,:].T
    X_flat=np.array([window(seg) for segs in all_segments for seg in segs])
    if fit:
        xdawn=Xdawn(nfilter=6)
        xdawn.fit(X_flat,y_all)
    X_per_char=[]
    for segs in all_segments:
        X_char=np.array([window(seg) for seg in segs])
        X_char=xdawn.transform(X_char)
        X_char=X_char.reshape(X_char.shape[0],-1)
        X_per_char.append(X_char)
    if fit:
        X_train=np.concatenate(X_per_char)
        return X_train,X_per_char
    return X_per_char