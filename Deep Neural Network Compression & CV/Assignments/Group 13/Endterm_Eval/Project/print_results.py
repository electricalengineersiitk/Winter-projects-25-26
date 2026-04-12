import torch
import os
import json
import torch.nn as nn
from pathlib import Path

# Imports from your project
from models.model_cifar import SmallCIFARNet
from data.data_loader import get_dataloaders
from utils.test_eval import evaluate
from utils.loading import load_model_from_npz
from utils.huffman import huffman_encode, get_relative_indices
from utils.performance import measure_performance
from config import config_device

# Reusing existing reporting functions

def get_disk_size(file_path):
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"Disk Space (.npz): {size_mb:.2f} MB")
        return size_mb
    return 0

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
    return total_params

def run_huffman_theoretical(model):
    print("\nRunning Stage 3: Huffman Coding...")
    total_huffman_bits = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            weights = module.weight.data
            mask = (weights != 0)
            
            if mask.sum() == 0: continue
            
            # 1. Encode weights (assuming 5-bit/32-cluster quantization)
            non_zero_weights = weights[mask].cpu().numpy().flatten()
            encoded_w, _ = huffman_encode(non_zero_weights.tolist())
            total_huffman_bits += len(encoded_w)
            
            # 2. Encode the Relative Indices
            rel_indices = get_relative_indices(mask.cpu().numpy())
            encoded_idx, _ = huffman_encode(rel_indices)
            total_huffman_bits += len(encoded_idx)

    huffman_mb = total_huffman_bits / (8 * 1024 * 1024)
    
    if huffman_mb == 0:
        huffman_mb = os.path.getsize("compressed_models/compressed.npz") / (1024 * 1024)
        
    print(f"Huffman Coding Theoretical Size: {huffman_mb:.4f} MB")
    return huffman_mb

def print_final_metrics(baseline_acc, final_acc, original_mb, compressed_mb, total_params):
    print("\n" + "="*30)
    print("FINAL EVALUATION METRICS")
    print("="*30)
    comp_ratio = original_mb / compressed_mb if compressed_mb > 0 else 0
    print(f"Compression Ratio: {comp_ratio:.2f}x")
    
    acc_change = final_acc - baseline_acc
    sign = "+" if acc_change > 0 else ""
    print(f"Accuracy Change: {sign}{acc_change:.2f}%")
    
    total_bits = compressed_mb * 1024 * 1024 * 8
    avg_bits = total_bits / total_params if total_params > 0 else 0
    print(f"Avg Bits per Weight: {avg_bits:.2f} bits")
    print("="*30)

def main():
    device = config_device()
    _, test_loader = get_dataloaders()
    model = SmallCIFARNet().to(device)
    
    npz_path = "compressed_models/compressed.npz"
    meta_path = "compressed_models/metadata.json"

    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found. Run main.py or compress.py first.")
        return

    # 1. Load the Compressed Weights
    load_model_from_npz(model, npz_path, device)
    model.eval()

    # 2. Load Baseline Accuracy (for the comparison metric)
    baseline_acc = 81.17  # Default if file missing
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            baseline_acc = json.load(f).get("baseline_accuracy", 81.17)

    # 3. Perform Evaluation
    print("Evaluating loaded model...")
    _, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss())
    
    # 4. Run Reports
    huffman_mb = run_huffman_theoretical(model)
    latency, ram_usage = measure_performance(model, device)
    
    print("\n" + "="*30)
    print("FINAL PROJECT REPORTS")
    print("="*30)
    get_disk_size(npz_path)
    print(f"Theoretical Huffman Size: {huffman_mb:.2f} MB")
    print(f"Inference Latency: {latency:.4f} ms/image")
    print(f"Peak RAM Usage: {ram_usage:.2f} MB")
    total_params = get_model_parameters(model)
    print("="*30)

    # Calculate dense size (4 bytes per float32)
    original_mb = (total_params * 4) / (1024 * 1024)
    print_final_metrics(baseline_acc, test_acc, original_mb, huffman_mb, total_params)

if __name__ == '__main__':
    main()