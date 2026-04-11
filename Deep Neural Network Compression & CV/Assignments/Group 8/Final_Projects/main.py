
import torch
import torch.nn as nn
import torch.optim as optim

from data.data_loader import get_data_loaders
from models.mnist import mnist
from utils.training import train, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_data_loaders()

# Model
model = mnist().to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(5):
    loss = train(model, train_loader, optimizer, criterion, device)
    acc = evaluate(model, test_loader, device)

    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {acc:.2f}%")




from utils.metrics import calculate_sparsity, quantization_report

def detailed_sparsity_report(model):
    sparsity = calculate_sparsity(model)
    print(f"Sparsity: {sparsity:.2f}%")

print("Baseline Accuracy:", evaluate(model, test_loader, device))


# Apply Pruning
PRUNE_RATIO = 0.5
model.prune(PRUNE_RATIO)
sparsity = calculate_sparsity(model)
print(f"Model Sparsity: {sparsity:.2f}%")




# Fine Tuning after Pruning
for epoch in range(3):
    loss = train(model, train_loader, optimizer, criterion, device)
    acc = evaluate(model, test_loader, device)

    print(f"[Finetune] Epoch {epoch+1}, Loss: {loss:.4f}, Acc: {acc:.2f}%")

print("Before Quantization Accuracy:", evaluate(model, test_loader, device))


# APPLY QUANTIZATION
NUM_BITS = 8

model.quantize(NUM_BITS)

print("After Quantization Accuracy:", evaluate(model, test_loader, device))




from utils.printing import print_header

print_header("Deep Compression Pipeline v1.0")

print("[*] Computation Device:", device)

# -----------------------
print_header("PHASE 1: BASELINE TRAINING")
# -----------------------

# training logs already there

# -----------------------
print_header("PHASE 2: PRUNING")
# -----------------------

print("--- BEFORE PRUNING ---")
detailed_sparsity_report(model)

model.prune(0.5)

print("--- AFTER PRUNING ---")
detailed_sparsity_report(model)

# -----------------------
print_header("PHASE 3: QUANTIZATION")
# -----------------------

print("[*] Before Quantization:")
quantization_report(model)

model.quantize(8)

print("[*] After Quantization:")
quantization_report(model)

# -----------------------
print_header("PHASE 4: HUFFMAN ENCODING")
# -----------------------

# apply_huffman_to_npz("compressed_models/model.npz")





# Here we Check compression effect
from utils.metrics import count_unique_weights
unique_vals = count_unique_weights(model)
print("Unique weight values:", unique_vals)

for module in model.modules():
    print(type(module))




def count_params(model):
    return sum(p.numel() for p in model.parameters())

print("Total Parameters:", count_params(model))





import psutil
import os
process = psutil.Process(os.getpid())
print("RAM (MB):", process.memory_info().rss / 1024**2)





