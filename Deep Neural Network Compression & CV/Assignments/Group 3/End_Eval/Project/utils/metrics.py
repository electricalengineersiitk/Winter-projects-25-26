## evaluation and compression measurement utilities
## functions to check accuracy, count parameters, measure sparsity, and estimate the final compressed model size

import torch
import torch.nn as nn
import numpy as np
import os


def evaluate_accuracy(model, data_loader, device): # runs the model on the test set and returns top 1 accuracy as a percentage
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    model.train()
    return accuracy


def count_parameters(model): # counts total and nonzero params, gives perlayer breakdown
    layer_stats = []
    total_all = 0
    nonzero_all = 0

    for name, param in model.named_parameters():
        total = param.numel()
        nonzero = (param.data != 0).sum().item()
        sparsity = 1.0 - (nonzero / total) if total > 0 else 0

        total_all += total
        nonzero_all += nonzero
        layer_stats.append((name, total, nonzero, sparsity * 100))

    return total_all, nonzero_all, layer_stats


def compute_sparsity(model): # fraction of weights that are zero across the whole model
    total, nonzero, _ = count_parameters(model)
    return 1.0 - (nonzero / total) if total > 0 else 0


def model_size_bytes(model): # raw model size assuming all weights are 32 bit floats
    total = sum(p.numel() for p in model.parameters())
    return total * 4


def compressed_size_bytes(model, codebooks=None, huffman_results=None):
    ## estimates compressed model size using the papers methodology
    ## weight indices (log2(k) bits each) + sparse indices + codebook overhead
    weight_bits = 0
    index_bits = 0
    codebook_bits = 0

    for name, param in model.named_parameters():
        if 'weight' in name and ('features' in name or 'classifier' in name):
            if 'bn' not in name and len(param.shape) >= 2:
                nonzero = (param.data != 0).sum().item()

                if codebooks and name in codebooks:
                    cb = codebooks[name]
                    k = cb['num_clusters']
                    bits_per_weight = cb['bits']

                    # uses huffman average bits if available, otherwise fixed length
                    if huffman_results and name in huffman_results:
                        hr = huffman_results[name]
                        w_bits = hr['huffman_bits_total']
                    else:
                        w_bits = nonzero * bits_per_weight

                    weight_bits += w_bits
                    codebook_bits += k * 32

                    # sparse index bits 8 for conv, 5 for fc
                    if 'features' in name:
                        idx_bits_per = 8
                    else:
                        idx_bits_per = 5
                    index_bits += nonzero * idx_bits_per

                else:
                    weight_bits += nonzero * 32

    total_bits = weight_bits + index_bits + codebook_bits

    breakdown = {
        'weight_bits': weight_bits,
        'index_bits': index_bits,
        'codebook_bits': codebook_bits,
        'total_bits': total_bits,
        'total_bytes': total_bits / 8,
        'total_kb': total_bits / 8 / 1024,
        'total_mb': total_bits / 8 / 1024 / 1024,
    }

    return total_bits, breakdown


def print_compression_summary(model, original_size_bytes, codebooks=None,
                               huffman_results=None, original_acc=None,
                               compressed_acc=None): # prints a full summary table showing the compression results
    print(f"\n{'='*80}")
    print(f"  COMPRESSION SUMMARY (Deep Compression Pipeline)")
    print(f"{'='*80}")

    original_mb = original_size_bytes / 1024 / 1024
    _, breakdown = compressed_size_bytes(model, codebooks, huffman_results)
    compressed_mb = breakdown['total_mb']

    compression_ratio = original_mb / compressed_mb if compressed_mb > 0 else float('inf')

    print(f"\n  Original model size:     {original_mb:.2f} MB")
    print(f"  Compressed model size:   {compressed_mb:.2f} MB")
    print(f"  Compression ratio:       {compression_ratio:.1f}x")

    if original_acc is not None and compressed_acc is not None:
        acc_drop = original_acc - compressed_acc
        print(f"\n  Original accuracy:       {original_acc:.2f}%")
        print(f"  Compressed accuracy:     {compressed_acc:.2f}%")
        print(f"  Accuracy drop:           {acc_drop:.2f}%")

    print(f"\n  Storage breakdown:")
    print(f"    Weight indices:  {breakdown['weight_bits']:12,} bits  "
          f"({breakdown['weight_bits']/breakdown['total_bits']*100:.1f}%)")
    print(f"    Sparse indices:  {breakdown['index_bits']:12,} bits  "
          f"({breakdown['index_bits']/breakdown['total_bits']*100:.1f}%)")
    print(f"    Codebooks:       {breakdown['codebook_bits']:12,} bits  "
          f"({breakdown['codebook_bits']/breakdown['total_bits']*100:.1f}%)")
    print(f"    Total:           {breakdown['total_bits']:12,} bits  "
          f"({breakdown['total_kb']:.1f} KB)")

    # per-layer table
    print(f"\n  {'Layer':<40s}  {'#Params':>8s}  {'NonZero':>8s}  "
          f"{'Sparsity':>8s}  {'Kept%':>6s}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")

    total_params, total_nonzero, layer_stats = count_parameters(model)
    for name, total, nonzero, sparsity in layer_stats:
        if 'weight' in name and len(name) > 0:
            kept_pct = (1 - sparsity / 100) * 100
            print(f"  {name:<40s}  {total:8d}  {nonzero:8d}  "
                  f"{sparsity:7.1f}%  {kept_pct:5.1f}%")

    overall_sparsity = (1 - total_nonzero / total_params) * 100 if total_params > 0 else 0
    print(f"\n  Overall: {total_params:,} params, {total_nonzero:,} nonzero, "
          f"{overall_sparsity:.1f}% sparse")
    print(f"{'='*80}\n")
