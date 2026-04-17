import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Baseline Training 
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 5


# Compression parameters
PRUNING_THRESHOLD = 0.5  # This will remove 20% of the weakest weights
QUANTIZATION_BITS = 8    # We'll need this later for Stage 2