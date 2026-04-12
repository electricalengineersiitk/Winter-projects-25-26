import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from skorch import NeuralNetClassifier
import torch.optim as optim

try:
    from pyriemann.classification import MDM
except ImportError:
    MDM = None

class EEGNet(nn.Module):
    def __init__(self, n_chan=16, n_time=32):
        super(EEGNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 8, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (n_chan, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.25)
        )
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
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis())
    ])
def get_svm_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, kernel='rbf'))
    ])
def get_eegnet_pipeline():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight = torch.tensor([1.0, 5.0], device=device)
    return NeuralNetClassifier(
        EEGNet,
        criterion=nn.CrossEntropyLoss,
        criterion__weight=weight,
        optimizer=optim.Adam,
        lr=0.001,
        max_epochs=50,
        batch_size=32,
        device=device,
        iterator_train__shuffle=True,
        train_split=None, 
        verbose=0
    )

def get_riemannian_pipeline():
    """
    Returns a Minimum Distance to Mean (MDM) classifier.
    Expects Covariance matrices as input.
    """
    if MDM is None:
        raise ImportError("pyriemann is not installed.")
    return MDM()
