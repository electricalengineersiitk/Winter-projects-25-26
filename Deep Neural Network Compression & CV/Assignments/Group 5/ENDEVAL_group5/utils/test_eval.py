# utils/test_eval.py — Evaluation and compression reporting

import torch
import os


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns accuracy as a float in [0, 1]."""
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds    = model(imgs).argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
    return correct / total


def print_compression_report(baseline_acc, pruned_acc, quant_acc,
                              reloaded_acc, orig_kb, quant_stats,
                              huff_stats, archive_path):
    """Prints the full 3-stage compression summary table."""

    archive_kb    = os.path.getsize(archive_path) / 1024
    overall_ratio = orig_kb / max(archive_kb, 1e-6)

    print('\n' + '═' * 72)
    print('  FULL 3-STAGE COMPRESSION REPORT')
    print('═' * 72)
    print(f"  {'Stage':<30} {'Accuracy':>10} {'Size (KB)':>12} {'Ratio':>8}")
    print('  ' + '─' * 64)
    print(f"  {'Original (baseline)':<30} "
          f"{baseline_acc*100:>9.2f}% {orig_kb:>11.2f}  {'1.00x':>8}")
    print(f"  {'After L1 Pruning (90%)':<30} "
          f"{pruned_acc*100:>9.2f}% {'(sparse)':>11}  {'—':>8}")
    print(f"  {'After K-Means Quant.':<30} "
          f"{quant_acc*100:>9.2f}% "
          f"{quant_stats['compressed_kb']:>11.2f}  "
          f"{quant_stats['overall_ratio']:>6.2f}x")
    print(f"  {'After Huffman Coding':<30} "
          f"{reloaded_acc*100:>9.2f}% "
          f"{archive_kb:>11.2f}  {overall_ratio:>6.2f}x")
    print('  ' + '─' * 64)
    print(f"  Total accuracy drop     : {(baseline_acc - reloaded_acc)*100:.2f}%")
    print(f"  Avg bits per weight     : {quant_stats['avg_bits_weight']:.3f}")
    print(f"  Huffman extra gain      : {huff_stats['huffman_ratio']:.4f}x")
    print('═' * 72 + '\n')

    print(f"  {'Layer':<40} {'k':>4} {'FixBits':>9} "
          f"{'HuffBits':>9} {'Gain':>7} {'AvgLen':>7}")
    print('  ' + '─' * 76)
    for l in huff_stats['per_layer']:
        print(f"  {l['name']:<40} {l['k']:>4} "
              f"{l['idx_bits_fixed']:>9,} {l['idx_bits_huffman']:>9,} "
              f"{l['layer_huff_ratio']:>6.3f}x {l['avg_code_length']:>7.3f}")
    print('═' * 72)
