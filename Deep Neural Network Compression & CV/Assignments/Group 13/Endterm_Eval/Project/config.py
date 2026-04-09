import torch
import os
from pathlib import Path

def config_device():
    device = None
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    #print(f"Using device: {device}")
    return device

# Automatically configure and store the device when config is imported
DEVICE = config_device()

# Data Settings
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / 'data' / 'data_files'

BATCH_SIZE = 64
NUM_WORKERS = 2

# Training Hyperparameters
LEARNING_RATE = 0.001
FINETUNE_LR = 0.0001
CENTROID_LR = 0.00005

EPOCHS = 10

# Compression Targets 
TARGET_SPARSITY = 0.90
K_MEANS_CLUSTERS = 32 