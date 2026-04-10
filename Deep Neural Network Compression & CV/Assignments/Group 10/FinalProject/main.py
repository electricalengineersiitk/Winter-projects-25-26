import torch
import numpy as np
from models import mnist_model  
from data import MNIST_loader  
from utils.training import train_model, evaluate, train_and_eval
from utils.test_eval import measure_model_size, measure_runtime_memory
from compression.prune import prune_model
from compression.quantization import quantize_model
from utils.loading import load_csr_from_npz, save_model_npz, load_model_from_npz
from config import config_device
from utils.test_eval import measure_model_size, measure_runtime_memory, measure_system_ram

path = 'data' 
device = config_device()


train_loader, test_loader = MNIST_loader(path)
model = mnist_model()
model = model.to(device)

train_and_eval(model, train_loader, test_loader, device, epochs=5)
prune_model(model, 0.98)
train_and_eval(model, train_loader, test_loader, device, epochs=5)

def count_mask_sparsity(model):
    total = 0
    zeros = 0
    for module in model.modules():
        if hasattr(module, 'mode') and module.mode == 'prune':
            if module.mask is not None:
                total += module.mask.numel()
                zeros += (module.mask == 0).sum().item()
    if total == 0:
        print("Sparsity Check: No pruned layers found.")
    else:
        print(f"Masked sparsity: {100 * zeros / total:.2f}%")

count_mask_sparsity(model)

torch.save(model.state_dict(), "compressed_models/model.pth")

quantize_model(model, 4) 
print("\n Model quantized to 4 clusters!")
train_and_eval(model, train_loader, test_loader, device, epochs=5)

npz_path = "compressed_models/compressed.npz"
save_model_npz(model, npz_path)

model2 = mnist_model()
model2 = load_model_from_npz(model2, npz_path, device)
train_loader, test_loader = MNIST_loader(path)
model2 = model2.to(device)

print(f"Compressed Model Accuracy: {evaluate(model2, test_loader, device):.2f}%")



print("\n--- Final Project Metrics ---")


size_normal = measure_model_size("compressed_models/model.pth") 
size_compressed = measure_model_size("compressed_models/compressed.npz") 


if size_compressed > 0: 
    ratio = size_normal / size_compressed
    print(f"\n SUCCESS: The model was compressed by {ratio:.2f}x!")
measure_runtime_memory(device)
measure_system_ram()