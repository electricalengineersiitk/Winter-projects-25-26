import torch
import numpy as np
import os
import math
import psutil
from tqdm import tqdm

TARGET_ACCURACY_DROP = 1.5
TARGET_BITS_PER_WEIGHT = 3.57


def get_npz_size_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def get_ram_usage_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    return total, trainable


def count_sparsity_ratio(model):
    from compression.conv2d import modified_conv2d
    from compression.linear import modified_linear
    total = zeros = 0
    for m in model.modules():
        if isinstance(m, (modified_conv2d, modified_linear)):
            total += m.mask.numel()
            zeros += (m.mask == 0).sum().item()
    return zeros, total


def test_eval(model, test_loader, device):
    model.to(device)
    model.eval()
    correct = total = 0
    loop = tqdm(test_loader, desc="Evaluating", leave=True, ncols=100)
    with torch.no_grad():
        for batch in loop:
            labels = batch["label"].to(device)
            batch_in = {k: v.to(device) for k, v in batch.items()
                        if k != "label"}
            preds = model(batch_in).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix({"acc": f"{100.0*correct/total:.2f}%"})
    return 100.0 * correct / total


def print_compression_table(
    baseline_npz, pruned_npz, quantized_npz, huffman_npz,
    baseline_acc,
    pruned_acc_before_retrain,
    pruned_acc_after_retrain,
    quantized_acc_before_retrain,
    quantized_acc_after_retrain,
    baseline_model, pruned_model, quantized_model,
    k=16
):
    baseline_mb = get_npz_size_mb(baseline_npz)
    pruned_mb = get_npz_size_mb(pruned_npz)
    quantized_mb = get_npz_size_mb(quantized_npz)
    huffman_mb = get_npz_size_mb(huffman_npz)

    def pct(mb):
        return 100 * (1 - mb / baseline_mb) if baseline_mb > 0 else 0

    def ratio(mb):
        return baseline_mb / mb if mb > 0 else 0

    zeros, total_w = count_sparsity_ratio(pruned_model)
    sparsity_pct = 100.0 * zeros / total_w if total_w > 0 else 0
    total_params, _ = count_parameters(baseline_model)
    vram_mb = get_gpu_memory_mb()
    ram_mb = get_ram_usage_mb()
    bits_quantized = math.log2(k) * (1 - sparsity_pct / 100)

    try:
        huff_data = np.load(huffman_npz, allow_pickle=True)
        bits_huffman = float(huff_data["avg_bits_per_weight"])
        huff_ratio = float(huff_data["huffman_ratio"])
    except Exception:
        bits_huffman = bits_quantized
        huff_ratio = 1.0

    div = "-" * 60

    def row(label, value):
        return f"  {label:<38}{value}"

    print("\n" + div)
    print(f"  DEEP COMPRESSION RESULTS")
    print(div)

    print(f"\n  1. ACCURACY")
    print(row("Baseline",
              f"{baseline_acc:.2f}%"))
    print(row("After Pruning  (before retrain)",
              f"{pruned_acc_before_retrain:.2f}%"
              f"   (loss {baseline_acc - pruned_acc_before_retrain:.2f}%)"))
    print(row("After Pruning  (recovered)",
              f"{pruned_acc_after_retrain:.2f}%"
              f"   (Δ {pruned_acc_after_retrain - baseline_acc:+.2f}%)"))
    print(row("After Quantization (before retrain)",
              f"{quantized_acc_before_retrain:.2f}%"
              f"   (loss "
              f"{pruned_acc_after_retrain - quantized_acc_before_retrain:.2f}%)"))
    print(row("After Quantization (recovered)",
              f"{quantized_acc_after_retrain:.2f}%"
              f"   (Δ {quantized_acc_after_retrain - baseline_acc:+.2f}%)"))
    total_drop = baseline_acc - quantized_acc_after_retrain
    print(row("Total Accuracy Drop",
              f"{total_drop:.2f}%  "))

    print(f"\n  2. STORAGE MEMORY")
    print(row("Baseline (.npz)",
              f"{baseline_mb:.2f} MB"))
    print(row("After Pruning (.npz)",
              f"{pruned_mb:.2f} MB"
              f"   ({pct(pruned_mb):.1f}% smaller  {ratio(pruned_mb):.1f}x)"))
    print(row("After Quantization (.npz)",
              f"{quantized_mb:.2f} MB"
              f"   ({pct(quantized_mb):.1f}% smaller  {ratio(quantized_mb):.1f}x)"))
    print(row("Huffman Compressed (.npz)",
              f"{huffman_mb:.2f} MB"
              f"   ({pct(huffman_mb):.1f}% smaller  {ratio(huffman_mb):.1f}x)"))
    print(row("Final Compression Ratio",
              f"{ratio(huffman_mb):.1f}x  "))

    print(f"\n  3. RUNTIME MEMORY")
    print(row("GPU VRAM Used",           f"{vram_mb:.1f} MB"))
    print(row("RAM Used by Process",     f"{ram_mb:.1f} MB"))
    print(row("Total Parameters",        f"{total_params:,}"))
    print(row("Weights Pruned",
              f"{zeros:,} / {total_w:,}   ({sparsity_pct:.1f}% sparse)"))

    print(f"\n  4. COMPRESSION DETAILS")
    print(row("Quantization",
              f"k={k} centroids   (~{int(math.log2(k))}-bit)"))
    print(row("Bits/weight after pruning+quantization",
              f"{bits_quantized:.4f}"))
    print(row("Bits/weight after Huffman",
              f"{bits_huffman:.4f}   "
              f"  (target {TARGET_BITS_PER_WEIGHT})"))
    print(row("Huffman coding gain",
              f"{huff_ratio:.2f}x over quantized"))

    print("\n" + div + "\n")
