import torch
import os
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path

# Imports
from models.model_cifar import SmallCIFARNet
from data.data_loader import get_dataloaders
from utils.training import train_one_epoch, train_quantized_epoch
from utils.test_eval import evaluate
from compression.prune import prune_model
from compression.quantization import quantize_model
from utils.loading import save_model_npz, load_model_from_npz
from config import config_device, TARGET_SPARSITY, K_MEANS_CLUSTERS, FINETUNE_LR, EPOCHS
from utils.huffman import huffman_encode, get_relative_indices
from utils.performance import measure_performance


# Definitions

def get_disk_size(file_path):
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        print(f"Disk Space: {size_mb:.2f} MB")
    else:
        print(f"File {file_path} not found for disk report.")

def get_model_parameters(model):
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()
            
    sparsity = (zero_params / total_params) * 100 if total_params > 0 else 0
    print(f"Total Parameters: {total_params:,}")
    print(f"Zero Parameters (Pruned): {zero_params:,}")
    print(f"Actual Sparsity: {sparsity:.2f}%")

def run_huffman_stage(model):
    print("\nRunning Stage 3: Huffman Coding...")
    total_huffman_bits = 0
    
    for name, m in model.named_modules():
        if hasattr(m, 'is_quantized') and m.is_quantized:
            # 1. Encode the Quantized Weights (Centroid Indices) [cite: 152]
            weights = m.weight.data.cpu().numpy()
            mask = m.mask.cpu().numpy().astype(bool)
            centroid_indices = m.indices.cpu().numpy()[mask].tolist()
            
            encoded_w, codebook_w = huffman_encode(centroid_indices)
            total_huffman_bits += len(encoded_w)
            
            # 2. Encode the Sparse Index Differences 
            rel_indices = get_relative_indices(mask)
            encoded_idx, codebook_idx = huffman_encode(rel_indices)
            total_huffman_bits += len(encoded_idx)
            
            # 3. Codebook overhead is usually negligible [cite: 264, 490]
            # (In a real implementation, you'd save the codebook too)

    huffman_mb = total_huffman_bits / (8 * 1024 * 1024)
    print(f"Huffman Coding Theoretical Size: {huffman_mb:.4f} MB")
    return huffman_mb

def print_final_metrics(baseline_acc, final_acc, original_size_mb, compressed_size_mb, total_params):
    print("\n" + "="*30)
    print("FINAL EVALUATION METRICS")
    print("="*30)
    
    # 1. Compression Ratio
    comp_ratio = original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 0
    print(f"Compression Ratio: {comp_ratio:.2f}x")
    
    # 2. Accuracy Loss/Gain
    acc_change = final_acc - baseline_acc
    sign = "+" if acc_change > 0 else ""
    word = "Gain" if acc_change > 0 else "Loss"
    print(f"Accuracy Change: {sign}{acc_change:.2f}% ({word})")
    
    # 3. Average Bits Per Weight
    # Convert compressed size in MB to bits
    total_bits = compressed_size_mb * 1024 * 1024 * 8
    avg_bits = total_bits / total_params if total_params > 0 else 0
    print(f"Avg Bits per Weight: {avg_bits:.2f} bits")
    print("="*30)

# Execution (Must be inside main)
def main():
    # 1. Setup Environment
    device = config_device()
    train_loader, test_loader = get_dataloaders()

    # 2. Initialize Model
    model = SmallCIFARNet().to(device)
    criterion = nn.CrossEntropyLoss()

   # STAGE 0: Initial Baseline
    print("\nTraining Initial Baseline...")
    optimizer = optim.Adam(model.parameters()) 
    for epoch in range(1, EPOCHS + 1): 
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch}: Acc {train_acc:.2f}%")
        
    # Save these for the final report
    baseline_accuracy = train_acc
    total_parameters = sum(p.numel() for p in model.parameters())
    # 32-bit float = 4 bytes per parameter
    original_mb = (total_parameters * 4) / (1024 * 1024)

    # STAGE 1: Pruning
    print(f"\nApplying Pruning ({TARGET_SPARSITY * 100}%)...")
    prune_model(model, TARGET_SPARSITY)

    print("Fine-tuning after Pruning...")
    optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR) 
    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Fine-tune Epoch {epoch}: Acc {acc:.2f}%")

    # STAGE 2: Quantization
    print(f"\nApplying K-Means Quantization ({K_MEANS_CLUSTERS} clusters)...")
    quantize_model(model, K_MEANS_CLUSTERS)

    # Stage 2.5: Trained Quantization (Fine-tuning Centroids)
    print("\nFine-tuning Quantized Centroids...")
    for epoch in range(1, 3):
        loss, acc = train_quantized_epoch(model, train_loader, criterion)
        print(f"Centroid Fine-tune Epoch {epoch}: Acc {acc:.2f}%")

    # Final test check
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # STAGE 3: Saving & Reporting
    os.makedirs("compressed_models", exist_ok=True)
    torch.save(model.state_dict(), "compressed_models/model_weights.pth")
    npz_path = "compressed_models/compressed.npz"
    save_model_npz(model, npz_path)

    # Calculate theoretical Huffman gain
    huffman_mb = run_huffman_stage(model)
    
    # 4. Performance Benchmarking
    print("\nBenchmarking Model Performance...")
    latency, ram_usage = measure_performance(model, device)

    print("\n" + "="*30)
    print("FINAL PROJECT REPORTS")
    print("="*30)
    get_disk_size(npz_path)
    print(f"Theoretical Huffman Size: {huffman_mb:.2f} MB")
    print(f"Inference Latency: {latency:.4f} ms/image")
    print(f"Peak RAM Usage: {ram_usage:.2f} MB")
    get_model_parameters(model)
    print("="*30)

    print_final_metrics(baseline_accuracy, test_acc, original_mb, huffman_mb, total_parameters)
    with open("compressed_models/metadata.json", "w") as f:
        json.dump({"baseline_accuracy": baseline_accuracy}, f)

if __name__ == '__main__':
    main()
