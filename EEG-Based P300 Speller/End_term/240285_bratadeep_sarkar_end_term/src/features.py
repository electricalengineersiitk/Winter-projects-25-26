import numpy as np
def extract_p300_features(epochs_data, decimation_factor=1):
    if decimation_factor > 1:
        return epochs_data[:, :, ::decimation_factor].reshape(len(epochs_data), -1)
    return epochs_data.reshape(len(epochs_data), -1)
