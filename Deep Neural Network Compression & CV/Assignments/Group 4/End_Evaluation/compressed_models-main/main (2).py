import torch
from models.model_cifar import CompressionMLP
from data.data_loader import CIFAR10_loader
from utils.training import train_and_eval
from compression.pruning import prune_model
from compression.quantization import quantize_model
from compression.huffman import huffman_encode_model, print_compression_summary
from utils.loading import save_model_npz # ADDED THIS IMPORT
from config import config_device

device = config_device()
train_loader, test_loader = CIFAR10_loader()

# 1. Baseline Model
print("\n--- Training Baseline MLP ---")
model = CompressionMLP().to(device)
train_and_eval(model, train_loader, test_loader, device, epochs=5)

# 2. Pruning
print("\n--- Applying Pruning (90%) ---")
prune_model(model, 0.90)
print("Fine-tuning pruned model...")
train_and_eval(model, train_loader, test_loader, device, epochs=2)

# 3. Quantization
print("\n--- Applying Quantization (16 clusters) ---")
quantize_model(model, 16)
# Small finetune after quantization if needed
# train_and_eval(model, train_loader, test_loader, device, epochs=1) 

# 4. Huffman Encoding
print("\n--- Applying Huffman Encoding ---")
huffman_stats = huffman_encode_model(model)
print_compression_summary(model)

# 5. Serialization (Uncommented and Active)
print("\n--- Serializing Model to NPZ ---")
npz_path = "compressed_models/compressed_mlp.npz"
save_model_npz(model, npz_path)