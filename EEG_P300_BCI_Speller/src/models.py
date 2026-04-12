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

try:
    from braindecode.models import EEGNetv4
except ImportError:
    EEGNetv4 = None
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
def get_eegnet_pipeline(in_chans=16, input_window_samples=257):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight = torch.tensor([1.0, 5.0], device=device)
    if EEGNetv4 is None:
        raise ImportError("braindecode is not installed.")
    return NeuralNetClassifier(
        EEGNetv4,
        module__in_chans=in_chans,
        module__n_classes=2,
        module__input_window_samples=input_window_samples,
        module__final_conv_length='auto',
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
