import torch
import torch.nn as nn
import torch.optim as optim
import config
from models.model_cifar import SmallCIFARNet as mnist_model
from data.dataloader import get_dataloaders as MNIST_loader
from utils.training import train_model as train
from utils.test_eval import evaluate_model as evaluate
from compression.prune import prune_model
from compression.quantization import quantize_model
from utils.loading import save_model_npz, load_model_from_npz

device = config.config_device()
num_classes = config.NUM_CLASSES 

# 1.Loading Data
train_loader, test_loader = MNIST_loader(config.TRAIN_DIR, config.TEST_DIR, config.BATCH_SIZE)

model = mnist_model(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_and_eval(model, train_loader, test_loader, device, epochs=1):
    train(model, train_loader, criterion, optimizer, device, epochs=epochs)
    acc = evaluate(model, test_loader, device)
    print(f"Accuracy: {acc:.2f}%")

# 2.Trainig and pruning.
print(f"\n--- Initial Training ({num_classes} classes) ---")
train_and_eval(model, train_loader, test_loader, device, epochs=1)

print("\n--- Pruning (90% Percentile) ---")
prune_model(model, 90)
train_and_eval(model, train_loader, test_loader, device, epochs=1)

def count_mask_sparsity(model):
    total = 0
    zeros = 0
    for module in model.modules():
        if hasattr(module, "mask"):
            total += module.mask.numel()
            zeros += (module.mask == 0).sum().item()
    if total > 0:
        print(f"Masked sparsity: {100 * zeros / total:.2f}%")
    else:
        print("No masks found in model.")

count_mask_sparsity(model)
torch.save(model.state_dict(), "compressed_models/model.pth")

# 3.Quantize.
print("\n--- Quantizing (K=16) ---")
quantize_model(model, 16)
train_and_eval(model, train_loader, test_loader, device, epochs=1)

# 4.Huffman Encoding & Decoding.
npz_path = "compressed_models/compressed.npz"
save_model_npz(model, npz_path)

print("\n--- Validating Decoded Model ---")
model2 = mnist_model(num_classes=num_classes).to(device)
model2 = load_model_from_npz(model2, npz_path, device)

final_acc = evaluate(model2, test_loader, device)
print(f"Decoded Model Accuracy: {final_acc:.2f}%")

print("\n--- Weight Inspection ---")
print(f"Layer keys: {model.classifier[0].__dict__.keys()}")
print(f"Quantized Weight Sample: {model.classifier[0].weight[0][:5]}")