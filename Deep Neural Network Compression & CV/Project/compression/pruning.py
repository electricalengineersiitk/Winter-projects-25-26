import torch
from torch import nn

def prune_model(model, prune_fraction):
    assert 0.0 < prune_fraction < 1.0
    all_weights = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.detach().abs().flatten()
            all_weights.append(w)
    all_weights = torch.cat(all_weights)

    k = int(prune_fraction * all_weights.numel())
    threshold = torch.kthvalue(all_weights, k).values.item()
    model.prune(threshold)
    return 

def quantize_model(model,K):
    model.quantize(K)
    return