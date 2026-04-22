# config.py — All hyperparameters and paths in one place

import torch

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Paths ─────────────────────────────────────────────────────────────────────
WEIGHTS_PATH        = 'compressed_models/final_model_weights.pth'
PRUNED_PATH         = 'compressed_models/pruned_model_weights.pth'
QUANTIZED_PATH      = 'compressed_models/quantized_model.pth'
ARCHIVE_PATH        = 'compressed_models/huffman_encoded.pkl'
RESULTS_JSON_PATH   = 'compressed_models/compression_results.json'
DATA_DIR            = './data'

# ── Dataset ───────────────────────────────────────────────────────────────────
BATCH_SIZE_TRAIN    = 64
BATCH_SIZE_TEST     = 128
NUM_WORKERS         = 2
CLASSES             = ('plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')

# ── Training ──────────────────────────────────────────────────────────────────
NUM_EPOCHS          = 120
LEARNING_RATE       = 0.1
MOMENTUM            = 0.9
WEIGHT_DECAY        = 5e-4
LR_STEP_SIZE        = 30        # StepLR: drop LR every N epochs
LR_GAMMA            = 0.1       # StepLR: multiply LR by this factor

# ── Pruning ───────────────────────────────────────────────────────────────────
PRUNING_AMOUNT      = 0.90      # 90% sparsity target
FINE_TUNE_EPOCHS    = 40
FINE_TUNE_LR        = 0.001

# ── Quantization ──────────────────────────────────────────────────────────────
N_CLUSTERS_CONV     = 256       # K-Means clusters for Conv2d layers
N_CLUSTERS_FC       = 32        # K-Means clusters for Linear layers
