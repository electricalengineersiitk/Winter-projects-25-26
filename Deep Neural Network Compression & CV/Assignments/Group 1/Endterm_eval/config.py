# config for the deep compression pipeline
# all the hyperparams, paths, settings etc. are here so i dont scatter them everywhere

import os

# paths
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(BASE_DIR,'data')
MODEL_DIR=os.path.join(BASE_DIR,'compressed_models')

# feature extraction settings
FEATURE_DIM=512        # resnet gives 512-dim feature vectors
IMAGE_SIZE=(224,224)   # standard resnet input size
BATCH_SIZE=64

# mlp architecture
MLP_HIDDEN_LAYERS=[256,128,64]  # going from 512 -> 256 -> 128 -> 64 -> num_classes
MLP_DROPOUT=0.3                 # to prevent overfitting on small feature set
ACTIVATION='relu'

# training hyperparams
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4     # L2 regularization, similar to ridge regression from assignment 5
EPOCHS_BASELINE=40   # more epochs for baseline training to get good accuracy
EPOCHS_FINETUNE=40    # fewer epochs for fine-tuning after pruning

# pruning
PRUNE_THRESHOLD=0.1   # magnitude below which weights get zeroed out
PRUNE_SPARSITY=0.9    # targeting 90% sparsity (prune 90% of weights)

# quantization (k-means based, like in assignment 5 but on weights instead of matrix elements)
NUM_BITS=4            # 4-bit quantization
NUM_CLUSTERS=16       # 2^4 = 16 unique weight values per layer

# huffman encoding
HUFFMAN_ENABLED=True  # final stage of compression pipeline
