
"""
models.py — LDA, SVM baselines + EEGNet (TensorFlow)
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# LDA
def build_lda() -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lda',    LinearDiscriminantAnalysis(solver='svd'))
    ])

# SVM 
def build_svm(C: float = 1.0, kernel: str = 'rbf') -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(C=C, kernel=kernel, probability=True, class_weight='balanced'))
    ])

# EEGNet 
def build_eegnet(n_channels: int, n_times: int,
                 nb_classes: int = 2,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 dropout: float = 0.5):
    """
    EEGNet — Lawhern et al. 2018 (TensorFlow / Keras).
    Input shape : (batch, n_channels, n_times, 1) [channels_last]
    """
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import (
        Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
        BatchNormalization, AveragePooling2D, Dropout,
        Activation, Flatten, Dense
    )
    from tensorflow.keras.constraints import max_norm

    # 1.Update Input shape for channels_last
    inp = Input(shape=(n_channels, n_times, 1))

    # Block 1: Temporal convolution + Depthwise spatial filter
    x = Conv2D(F1, (1, 64), padding='same', use_bias=False)(inp)
    x = BatchNormalization(axis=-1)(x)
    x = DepthwiseConv2D((n_channels, 1), use_bias=False,
                        depth_multiplier=D,
                        depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropout)(x)

    # Block 2: Separable convolution 
    x = SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropout)(x)

    # Classifier head 
    x   = Flatten()(x)
    out = Dense(nb_classes, activation='softmax',
                kernel_constraint=max_norm(0.25))(x)

    model = Model(inputs=inp, outputs=out)
    return model