import torch
from braindecode.models import EEGNet
from braindecode import EEGClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def get_model(name="lda", n_chans=None, n_times=None):

    if name == "lda":
        return LinearDiscriminantAnalysis()

    if name == "svm":
        return CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000))

    if name == "eegnet":
        net = EEGNet(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=2
        )

        return EEGClassifier(
            net,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            train_split=None,
            batch_size=64,
            max_epochs=5,
            verbose=1
        )