import os
import random
import numpy as np
import torch
import kagglehub

from data import data_loader
from models import cifar_model
from compression import (prune_model, quantize_model,
                         count_sparsity, huffman_compress_model)
from utils import (train_and_eval, test_eval,
                   print_compression_table,
                   save_model_npz, load_model_from_npz)
from config import config_device

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"[Seed] Random seed set to {SEED}")

if __name__ == '__main__':

    path = kagglehub.dataset_download("moltean/fruits")
    path = os.path.join(path, "fruits-360_100x100", "fruits-360")
    training = os.path.join(path, "Training")
    test_dir = os.path.join(path, "Test")

    train_loader, test_loader = data_loader(training, test_dir)

    batch = next(iter(train_loader))
    print("\nBatch Shape Verification")
    for k, v in batch.items():
        print(f"{k:<20} {tuple(v.shape)}")

    num_classes = len(train_loader.dataset.class_to_idx)
    print(f"[Data] Number of classes: {num_classes}")

    device = config_device()
    os.makedirs("compressed_models", exist_ok=True)

    print("\nModel 1: Baseline Training")
    model = cifar_model(num_classes=num_classes).to(device)
    print(f"[Model] Parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    baseline_acc = train_and_eval(
        model, train_loader, test_loader, device, epochs=10
    )
    save_model_npz(model, "compressed_models/baseline.npz")

    print("\nModel 2: Pruning")
    prune_model(model, threshold=0.05)
    count_sparsity(model)

    print("\n  Evaluating accuracy loss from pruning...")
    pruned_acc_before = test_eval(model, test_loader, device)

    print("\n  Retraining after pruning...")
    pruned_acc_after = train_and_eval(
        model, train_loader, test_loader, device, epochs=5
    )
    save_model_npz(model, "compressed_models/pruned.npz")

    pruned_model_ref = cifar_model(num_classes=num_classes).to(device)
    pruned_model_ref = load_model_from_npz(
        pruned_model_ref, "compressed_models/pruned.npz", device
    )

    print("\nModel 3: Quantization on Pruned")
    quantize_model(model, k=16)

    print("\n  Evaluating accuracy loss from quantization...")
    quantized_acc_before = test_eval(model, test_loader, device)

    print("\n  Retraining after quantization...")
    quantized_acc_after = train_and_eval(
        model, train_loader, test_loader, device, epochs=5
    )
    save_model_npz(model, "compressed_models/quantized.npz")

    print("\nModel 4: Huffman Coding")
    huff_mb, weights_mb = huffman_compress_model(
        model, "compressed_models/huffman.npz"
    )

    baseline_model_ref = cifar_model(num_classes=num_classes).to(device)
    baseline_model_ref = load_model_from_npz(
        baseline_model_ref, "compressed_models/baseline.npz", device
    )

    print_compression_table(
        baseline_npz="compressed_models/baseline.npz",
        pruned_npz="compressed_models/pruned.npz",
        quantized_npz="compressed_models/quantized.npz",
        huffman_npz="compressed_models/huffman.npz",
        baseline_acc=baseline_acc,
        pruned_acc_before_retrain=pruned_acc_before,
        pruned_acc_after_retrain=pruned_acc_after,
        quantized_acc_before_retrain=quantized_acc_before,
        quantized_acc_after_retrain=quantized_acc_after,
        baseline_model=baseline_model_ref,
        pruned_model=pruned_model_ref,
        quantized_model=model,
        k=16
    )
