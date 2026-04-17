import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


from mne.decoding import Vectorizer, XdawnTransformer


def make_features_xdawn(epochs):

  m_f = make_pipeline(
      XdawnTransformer(n_components=3,reg=.1),
      Vectorizer(),
      MinMaxScaler()
  )

  X = epochs
  y = epochs.events[:,-1]

  m_f.fit(X,y)
  X = m_f.transform(X)

  return X



def make_features_downscale(epochs, target_samples=30):

    X = epochs.get_data()
    n_epochs, n_ch, n_times = X.shape
    if target_samples and target_samples < n_times:
        indices = np.linspace(0, n_times - 1, target_samples, dtype=int)
        X = X[:, :, indices]
    return X.reshape(n_epochs, -1)
  