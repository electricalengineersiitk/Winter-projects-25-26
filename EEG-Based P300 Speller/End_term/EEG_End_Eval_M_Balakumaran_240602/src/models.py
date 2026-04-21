from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D,
                                     SeparableConv2D, BatchNormalization,Activation,
                                     AveragePooling2D, Dropout, Flatten, Dense)
from tensorflow.keras.constraints import max_norm


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

def build_svm_classifier(kernel='linear', C=1.0, random_state=42, probability=True):
    return SVC(kernel=kernel, C=C, random_state=random_state, probability=probability)

def build_lda_classifier(solver='svd', shrinkage=None):
    return LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)


def EEGNet(nb_classes, Chans, Samples, F1=8, D=2, F2=16, dropoutRate=0.5):
    """
    EEGNet architecture.

    Parameters:
    - nb_classes: number of output classes
    - Chans: number of EEG channels
    - Samples: number of time samples per epoch
    - F1: number of temporal filters
    - D: depth multiplier for spatial filters
    - F2: number of pointwise filters
    """

    # Input is (batch, Chans, Samples, 1) for channels_last
    inputs = Input(shape=(Chans, Samples, 1))

    # Block 1: Temporal Convolution
    # Kernel (1, temporal_filter_length) operates on (height, width) where height=Chans (implicitly across channels) and width=Samples.
    # Using kernel (1, 120) for spatial (channel-wise) and temporal (time-wise) dimensions respectively.
    block1 = Conv2D(F1, (1, 120), padding = 'same', use_bias = False, data_format='channels_last')(inputs)
    block1 = BatchNormalization(axis=-1)(block1) # axis=-1 is the channel axis for channels_last

    # Block 1: Spatial Convolution
    # Kernel (Chans, 1) operates across channels (height) dimension.
    block1 = DepthwiseConv2D((Chans, 1), use_bias = False, depth_multiplier = D, depthwise_constraint = max_norm(1.), data_format='channels_last')(block1)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4), data_format='channels_last')(block1)
    block1 = Dropout(dropoutRate)(block1)

    # Block 2: Separable Convolution
    # Separable Conv for temporal features (similar to first Conv2D)
    block2 = SeparableConv2D(F2, (1, 16), use_bias = False, padding = 'same', data_format='channels_last')(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8), data_format='channels_last')(block2)
    block2 = Dropout(dropoutRate)(block2)

    # Classification
    flatten = Flatten(name = 'flatten')(block2)
    dense = Dense(nb_classes, name = 'dense')(flatten)
    softmax = Activation('softmax', name = 'softmax')(dense)

    return Model(inputs=inputs, outputs=softmax)


