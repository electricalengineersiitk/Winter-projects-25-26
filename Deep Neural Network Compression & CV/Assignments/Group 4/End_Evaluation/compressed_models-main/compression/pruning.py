import torch
from .linear import ModifiedLinear

def prune_model(model, prune_ratio=0.90):
    all_weights = []
    for module in model.modules():
        if isinstance(module, ModifiedLinear):
            all_weights.append(module.weight.data.abs().flatten())
            
    if not all_weights:
        return model
        
    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights, prune_ratio).item()
    
    total_weights, pruned_weights = 0, 0
    for module in model.modules():
        if isinstance(module, ModifiedLinear):
            module.prune(threshold)
            total_weights += module.mask.numel()
            pruned_weights += (module.mask == 0).sum().item()
            
    actual_sparsity = 100.0 * pruned_weights / total_weights if total_weights > 0 else 0
    print(f"Pruned: {pruned_weights}/{total_weights} ({actual_sparsity:.2f}% sparsity)")
    return model