import torch
import numpy as np

def prune_model(model, percentile):
    if percentile < 1.0:
        percentile = percentile * 100.0
        
    for name, module in model.named_modules():
        if hasattr(module, 'prune') and callable(getattr(module, 'prune')):
            weights_abs = torch.abs(module.weight.data).cpu().numpy()
            threshold = np.percentile(weights_abs, percentile)
            
            module.prune(threshold)