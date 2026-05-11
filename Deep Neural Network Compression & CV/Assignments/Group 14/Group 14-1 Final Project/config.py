import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRUNING_CYCLES = 10
PRUNE_RATE = 0.20
EPOCHS_PER_CYCLE = 5
BATCH_SIZE = 128
LR = 0.001