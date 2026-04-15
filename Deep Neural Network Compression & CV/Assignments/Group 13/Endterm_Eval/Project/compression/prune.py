import torch
import numpy as np
from compression.conv2d import modified_conv2d
from compression.linear import modified_linear

def prune_model(model, sparsity_target=0.9):
    """
    Implements L1 pruning to reach a global sparsity target.
    """
    all_weights = []
    
    # 1. Collect all weights from compressible layers
    for m in model.modules():
        if isinstance(m, (modified_conv2d, modified_linear)):
            all_weights.append(m.weight.data.abs().view(-1))
            
    # 2. Calculate the global threshold for the bottom 90%
    all_weights = torch.cat(all_weights)
    threshold = np.percentile(all_weights.cpu().numpy(), sparsity_target * 100)
    
    # 3. Apply the mask to each layer
    print(f"Pruning threshold: {threshold:.6f}")
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (modified_conv2d, modified_linear)):
                # Create a mask: 1 if weight > threshold, 0 otherwise
                mask = (m.weight.data.abs() > threshold).float()
                m.mask.copy_(mask) # Update the mask buffer in the layer
                m.weight.data.mul_(m.mask) # Apply it immediately
    
    print(f"Model pruned to {sparsity_target*100}% sparsity.")