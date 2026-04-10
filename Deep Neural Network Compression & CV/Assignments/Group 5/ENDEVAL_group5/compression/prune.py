# compression/prune.py — L1 Unstructured Pruning

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def apply_pruning(model, amount):
    """
    Applies L1 unstructured pruning to all Conv2d and Linear layers.

    Uses a binary mask — zeros out the lowest |amount| fraction of weights
    by absolute value. Masks are soft at this stage (reversible).

    Args:
        model  : nn.Module to prune (modified in-place)
        amount : float in (0, 1), fraction of weights to zero out
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)

    print(f'L1 pruning applied  ({amount*100:.0f}% sparsity target)')
    return model


def make_pruning_permanent(model):
    """
    Bakes the pruning masks into the weight tensors and removes mask buffers.
    Must be called before saving or passing to quantization.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass   # already permanent
    return model


def check_sparsity(model):
    """
    Prints per-layer and overall sparsity.
    Returns overall sparsity as a float in [0, 1].
    """
    total = zeros = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w      = module.weight.data
            z      = (w == 0).sum().item()
            n      = w.numel()
            zeros += z
            total += n
            print(f'  {name:<40} sparsity: {100.*z/n:5.1f}%')
    overall = zeros / total if total > 0 else 0.0
    print(f'  Overall sparsity: {overall*100:.2f}%')
    return overall
