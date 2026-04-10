import torch
import numpy as np

def prune_model(model, sparsity_ratio):

    
    all_weights = []
    
    for module in model.modules():
        if hasattr(module, 'prune') and hasattr(module, 'weight'):
            weights_flat = torch.abs(module.weight.data).cpu().numpy().flatten()
            all_weights.append(weights_flat)
            
    if not all_weights:
        print("Warning: No prunable layers found!")
        return

    all_weights_array = np.concatenate(all_weights)
    
    
    percentile_target = sparsity_ratio * 100
    global_threshold = np.percentile(all_weights_array, percentile_target)
    
    print(f"Target Sparsity: {percentile_target}% | Calculated Threshold: {global_threshold:.6f}")
    
   
    model.prune(global_threshold)
    
    return