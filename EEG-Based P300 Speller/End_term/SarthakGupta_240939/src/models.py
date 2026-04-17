import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import torch
from braindecode.models import EEGNet
from braindecode import EEGClassifier

MATRIX=np.array([
    list("ABCDEF"),
    list("GHIJKL"),
    list("MNOPQR"),
    list("STUVWX"),
    list("YZ1234"),
    list("56789_")
])

def train_model(X,y,model_type="lda",n_chans=None,n_times=None):
    if model_type=="lda":
        model=LinearDiscriminantAnalysis()
    elif model_type=="svm":
        model=SVC(kernel='rbf',probability=True,C=1.0,gamma='scale')
    elif model_type=="eegnet":
        eegnet=EEGNet(n_chans=n_chans,n_times=n_times,n_outputs=2)
        model=EEGClassifier(
            eegnet,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            train_split=None,
            batch_size=64,
            max_epochs=10,
            verbose=1
        )
    model.fit(X,y)
    return model

def decode_string(model,X_per_char,codes_per_char,n_reps=15):
    chars=[]
    for X_char,codes in zip(X_per_char,codes_per_char):

        cutoff=n_reps*12
        X_char=X_char[:cutoff]
        codes=codes[:cutoff]
        probs=model.predict_proba(X_char)[:,1]
        scores={}

        for p,c in zip(probs,codes):
            scores.setdefault(int(c),[]).append(p)

        avg={k:np.mean(v) for k,v in scores.items()}
        best_row=max(range(1,7),key=lambda x:avg.get(x,0))
        best_col=max(range(7,13),key=lambda x:avg.get(x,0))-7
        chars.append(MATRIX[best_col][best_row-1])
        
    return "".join(chars)