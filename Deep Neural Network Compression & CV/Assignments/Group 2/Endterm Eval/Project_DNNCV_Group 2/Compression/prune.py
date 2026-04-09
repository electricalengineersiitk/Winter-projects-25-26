
import numpy as np
import torch
from compression.linear import modified_linear
from compression.conv2d import modified_conv2d

def prune_model(model, percentile):
    """
    We have the percentile so we are calculating the ith percentile element 
    for each layer and we are going to call the layers own prune method to prune the layer.
    """
    for module in model.modules():
        if isinstance(module, (modified_linear, modified_conv2d)):
            
            weights = module.weight.data.abs().cpu().numpy().flatten()
            
            threshold = np.percentile(weights, percentile)
            
            module.prune(threshold)