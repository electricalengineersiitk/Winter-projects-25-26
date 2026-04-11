import os
import torch

from models.mnist            import mnist_model
from data.data_loader        import MNIST_loader
from utils.training          import train_and_eval, evaluate
from compression.pruning     import prune_model, quantize_model
from utils.loading           import load_csr_from_npz, save_model_npz, load_model_from_npz
from config                  import config_device


path   = "data"          
device = config_device()

os.makedirs("compressed_models", exist_ok=True)


train_loader, test_loader = MNIST_loader(path, batch_size=64)

model = mnist_model(num_classes=131, input_size=100).to(device)

train_and_eval(model, train_loader, test_loader, device, epochs=10)


prune_model(model, 0.90)
train_and_eval(model, train_loader, test_loader, device, epochs=5)


def count_mask_sparsity(model):
    total = zeros = 0
    for m in model.modules():
        if hasattr(m, "mask"):
            total += m.mask.numel()
            zeros += (m.mask == 0).sum().item()
    if total > 0:
        print(f"Masked sparsity: {100.0 * zeros / total:.2f}%")
    else:
        print("No masks found.")

count_mask_sparsity(model)


torch.save(model.state_dict(), "compressed_models/model.pth")
print(f"Pruned model saved  →  compressed_models/model.pth  "
      f"({os.path.getsize('compressed_models/model.pth') / 1024:.1f} KB)")


quantize_model(model, 16)
train_and_eval(model, train_loader, test_loader, device, epochs=1)


npz_path = "compressed_models/compressed.npz"
save_model_npz(model, npz_path)


criterion = torch.nn.CrossEntropyLoss()

model2 = mnist_model(num_classes=131, input_size=100)   
model2 = load_model_from_npz(model2, npz_path, device)  

print("\n── Compressed model evaluation ──")
print(evaluate(model2, test_loader, criterion, device))


def model_memory_mb(m):
    total = sum(
        p.numel() * p.element_size()
        for p in list(m.parameters()) + list(m.buffers())
    )
    return total / (1024 ** 2)

print("\n── Resource report ──────────────────────────────────────────────────")
print(f"  model.pth  (pruned float) : "
      f"{os.path.getsize('compressed_models/model.pth') / 1024:.1f} KB  on disk")
print(f"  compressed.npz (Huffman)  : "
      f"{os.path.getsize(npz_path) / 1024:.1f} KB  on disk")
print(f"  model2 in-memory          : {model_memory_mb(model2):.2f} MB")

if torch.cuda.is_available():
    print(f"  GPU memory allocated      : "
          f"{torch.cuda.memory_allocated(device) / (1024**2):.2f} MB")