import os
import torch

from utils.config import config_device
from utils.test_eval import train_and_eval, evaluate_only
from utils.loading import save_model_npz, load_model_from_npz
from utils.memory_profiler import report_runtime_memory

from data.data_loader import get_fruits_loaders
from models.model_fruits import FruitsFusionNet

from compression.prune import prune_model, count_mask_sparsity
from compression.quantization import quantize_model

def main():
    device = config_device()
    train_loader, test_loader, num_classes = get_fruits_loaders(batch_size=64)
    os.makedirs("compressed_models", exist_ok=True)

    print("\nBASELINE TRAINING\n")
    model = FruitsFusionNet(num_classes=num_classes).to(device)
    train_and_eval(model, train_loader, test_loader, device, epochs=15, lr=0.001)
    torch.save(model.state_dict(), "compressed_models/model.pth")

    print("\nMAGNITUDE PRUNING\n")
    prune_model(model, 0.50) 
    count_mask_sparsity(model)
    train_and_eval(model, train_loader, test_loader, device, epochs=5, lr=0.0005)

    print("\nK-MEANS QUANTIZATION\n")
    quantize_model(model, 16)
    train_and_eval(model, train_loader, test_loader, device, epochs=5, lr=0.0005)

    print("\nSAVING TO DISK\n")
    npz_path = "compressed_models/compressed.npz"
    save_model_npz(model, npz_path)

    print("\nDECOMPRESSION & EVALUATION\n")
    model2 = FruitsFusionNet(num_classes=num_classes)
    model2 = load_model_from_npz(model2, npz_path, device)
    model2.to(device)
    
    final_acc = evaluate_only(model2, test_loader, device)
    print(f"\nFinal Decompressed Model Accuracy: {final_acc:.2f}%")
    
    pth_size = os.path.getsize("compressed_models/model.pth") / (1024 * 1024)
    npz_size = os.path.getsize("compressed_models/compressed.npz") / (1024 * 1024)
    
    print("\nCOMPRESSION REPORT\n")
    print(f"Original PyTorch Disk Size: {pth_size:.2f} MB")
    print(f"Compressed NPZ Disk Size:   {npz_size:.2f} MB")
    print(f"Compression Ratio:          {pth_size / npz_size:.2f}x smaller")
    print("-" * 40)
    report_runtime_memory()

if __name__ == "__main__":
    main()
