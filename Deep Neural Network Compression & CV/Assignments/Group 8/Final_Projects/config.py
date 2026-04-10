# Configuration parameters for MLP model
DATA_DIR = "data/"
BATCH_SIZE = 128
NUM_EPOCHS = 30
FINETUNE_EPOCHS = 10
LR = 1e-3

PRUNING_RATIO = 0.5      # 50% weights are to be pruned
NUM_BITS = 8            

FEATURE_DIM = 512        
NUM_CLASSES = 10

SEED = 42
