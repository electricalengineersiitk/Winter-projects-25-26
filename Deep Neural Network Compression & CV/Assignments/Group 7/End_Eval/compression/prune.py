import torch
from .conv2d import modified_conv2d
from .linear import modified_linear


def prune_model(model, threshold=0.05):
    for module in model.modules():
        if isinstance(module, (modified_conv2d, modified_linear)):
            module.prune(threshold)
    print(f"[Pruning] Done — threshold={threshold}")


def count_sparsity(model):
    total = zeros = 0
    for module in model.modules():
        if isinstance(module, (modified_conv2d, modified_linear)):
            total += module.mask.numel()
            zeros += (module.mask == 0).sum().item()
    sparsity = 100.0 * zeros / total if total > 0 else 0.0
    print(f"[Sparsity] {zeros}/{total} weights pruned ({sparsity:.2f}%)")
    return zeros, total