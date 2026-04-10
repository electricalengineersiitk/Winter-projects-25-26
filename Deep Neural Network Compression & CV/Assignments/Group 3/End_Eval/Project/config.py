## all hyperparameters and paths for the deep compression pipeline
## based on the ICLR 2016 paper by Han et al.
## target: ~9x compression on a vgg-style cnn trained on cifar-10, with less than 1.5% accuracy drop

import os

# paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cifar10")
COMPRESSED_DIR = os.path.join(PROJECT_ROOT, "compressed_models")

# dataset
DATASET = "CIFAR-10"
NUM_CLASSES = 10
INPUT_SHAPE = (3, 32, 32)

# baseline training
BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LR_SCHEDULE_MILESTONES = [50, 75, 90]   # reduce lr at these epochs
LR_GAMMA = 0.1

# stage 1: pruning — remove 90% of the smallest weights
PRUNE_SPARSITY = 0.90
PRUNE_FINETUNE_EPOCHS = 25    # generous fine-tuning budget so the model can recover fully after losing 90% of weights
PRUNE_FINETUNE_LR = 0.001

# stage 2: quantization — 8-bit for conv (256 clusters), 5-bit for fc (32 clusters)
QUANT_BITS_CONV = 8
QUANT_BITS_FC = 5
QUANT_FINETUNE_EPOCHS = 15    # centroids need enough adjustment time to settle into values that preserve accuracy
QUANT_FINETUNE_LR = 0.001     # higher lr helps centroids converge faster during the limited fine-tuning window
QUANT_INIT_METHOD = "linear"  # linear initialization works best

# stage 3: huffman coding — no hyperparams needed, it's lossless and runs offline

# device
DEVICE = "cuda"   # falls back to cpu at runtime if cuda isn't available

# misc
SEED = 42
PRINT_EVERY = 50
