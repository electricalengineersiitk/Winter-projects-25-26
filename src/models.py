import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

class EEGNet(nn.Module):
    """Compact CNN for EEG classification (Lawhern et al., 2018)."""
    def __init__(self, n_chan=16, n_time=32):
        super(EEGNet, self).__init__()
        # Block 1: Temporal & Spatial Convolutions
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 8, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (n_chan, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.25)
        )
        # Block 2: Separable Convolutions
        self.b2 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 8), groups=16, padding='same', bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.25)
        )
        self.fc = nn.LazyLinear(2)

    def forward(self, x):
        return self.fc(self.b2(self.b1(x)).view(x.size(0), -1))

def get_lda_pipeline():
    """Standard Baseline: Scaler + LDA."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis())
    ])

def get_svm_pipeline():
    """Strong Classical Baseline: Scaler + RBF SVM."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, kernel='rbf'))
    ])

# Note: Xdawn is usually implemented as a spatial filter fit on train epochs.
# Since it takes Epochs as input (supervised), it's called manually in the CV loop 
# in evaluate.py but uses the LDA pipeline for classification.
