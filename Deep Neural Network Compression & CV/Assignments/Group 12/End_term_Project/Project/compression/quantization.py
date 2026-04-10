import torch

def quantize_model(model, K):

    for name, module in model.named_modules():
        if hasattr(module, 'quantize') and callable(getattr(module, 'quantize')):
            module.quantize(K)