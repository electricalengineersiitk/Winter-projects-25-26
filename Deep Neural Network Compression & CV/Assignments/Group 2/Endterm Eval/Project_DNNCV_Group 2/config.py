import torch
import os

def config_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

BATCH_SIZE = 64
TRAIN_DIR = '/kaggle/input/datasets/moltean/fruits/fruits-360_100x100/fruits-360/Training'
TEST_DIR = '/kaggle/input/datasets/moltean/fruits/fruits-360_100x100/fruits-360/Test'

if os.path.exists(TRAIN_DIR):
    NUM_CLASSES = len([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
else:
    NUM_CLASSES = 131
    
print(f"Config initialized with {NUM_CLASSES} classes.")