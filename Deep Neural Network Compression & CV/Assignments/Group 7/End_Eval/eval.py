import os
import sys
import math
import argparse
import numpy as np
import torch
import kagglehub

from data   import data_loader
from models import cifar_model
from utils  import load_model_from_npz, test_eval
from config import config_device


def get_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_sparsity(model):
    from compression.conv2d import modified_conv2d
    from compression.linear import modified_linear
    total = zeros = 0
    for m in model.modules():
        if isinstance(m, (modified_conv2d, modified_linear)):
            total += m.mask.numel()
            zeros += (m.mask == 0).sum().item()
    return zeros, total


def get_huffman_stats(path):
    data = np.load(path, allow_pickle=True)
    if "avg_bits_per_weight" in data.files:
        return (
            float(data["avg_bits_per_weight"]),
            float(data["huffman_ratio"]),
        )
    return None, None


def resolve_load_path(path):
    if os.path.basename(path) == "huffman.npz":
        weights_path = path.replace(".npz", "_weights.npz")
        if os.path.exists(weights_path):
            print(f"[Info] huffman.npz has no weights — "
                  f"loading from {os.path.basename(weights_path)}")
            return weights_path
        else:
            print(f"[Error] {weights_path} not found.")
            print("Please also download huffman_weights.npz from the Drive link.")
            sys.exit(1)
    return path


def print_eval_table(model, reported_path, load_path, accuracy, k=16):
    reported_mb  = get_size_mb(reported_path)
    load_mb      = get_size_mb(load_path)
    total_params = count_params(model)
    zeros, total = get_sparsity(model)
    sparsity_pct = 100.0 * zeros / total if total > 0 else 0
    active       = total - zeros
    bits_quant   = math.log2(k) * (1 - sparsity_pct / 100)

    is_huffman       = (reported_path != load_path)
    avg_bpw, h_ratio = get_huffman_stats(reported_path) \
                       if is_huffman else (None, None)

    div = "-" * 60

    def row(label, value):
        return f"  {label:<34}{value}"

    print("\n" + div)
    print(f"  EVALUATION RESULTS")
    print(f"  {os.path.basename(reported_path)}")
    print(div)

    print(f"\n  ACCURACY")
    print(row("Test Accuracy",              f"{accuracy:.2f}%"))

    print(f"\n  STORAGE")
    print(row("Compressed file size",       f"{reported_mb:.2f} MB"))
    if is_huffman:
        print(row("Loadable weights file",
                  f"{load_mb:.2f} MB   (huffman_weights.npz)"))

    print(f"\n  PARAMETERS")
    print(row("Total parameters",           f"{total_params:,}"))
    print(row("Non-zero (active)",
              f"{active:,}   ({100 - sparsity_pct:.1f}%)"))
    print(row("Pruned (zero)",
              f"{zeros:,}   ({sparsity_pct:.1f}% sparse)"))

    print(f"\n  COMPRESSION DETAILS")
    print(row("Quantization",
              f"k={k} centroids   (~{int(math.log2(k))}-bit)"))
    print(row("Bits/weight (pruning + quant)", f"{bits_quant:.4f}"))

    if avg_bpw is not None:
        print(row("Bits/weight (after Huffman)",
                  f"{avg_bpw:.4f}   "))
        print(row("Huffman gain",
                  f"{h_ratio:.2f}x over int16"))

    print("\n" + div + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a downloaded Deep Compression model"
    )
    parser.add_argument(
        "--model",
        type    = str,
        default = "DNN_Compression\compressed_models\pruned.npz",
        help    = "Path to .npz model file"
    )
    parser.add_argument(
        "--num_classes",
        type    = int,
        default = 257,
        help    = "Number of output classes (default: 257)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[Error] File not found: {args.model}")
        print("Download the model from the Drive links in the README.")
        sys.exit(1)

    device = config_device()

    print("\n[Setup] Downloading Fruits-360 dataset...")
    path     = kagglehub.dataset_download("moltean/fruits")
    path     = os.path.join(path, "fruits-360_100x100", "fruits-360")
    training = os.path.join(path, "Training")
    test_dir = os.path.join(path, "Test")

    _, test_loader = data_loader(training, test_dir)

    load_path = resolve_load_path(args.model)

    print(f"\n[Setup] Loading model from {load_path} ...")
    model = cifar_model(num_classes=args.num_classes).to(device)
    model = load_model_from_npz(model, load_path, device)

    print("\n[Eval] Running evaluation...")
    accuracy = test_eval(model, test_loader, device)

    print_eval_table(model, args.model, load_path, accuracy)


if __name__ == "__main__":
    main()