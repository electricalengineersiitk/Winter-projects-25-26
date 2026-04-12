
import torch
import torch.nn as nn
import torch.nn.functional as F

class modified_linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # mask for pruning
        self.register_buffer("mask", torch.ones_like(self.weight))

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)
    


    # Pruning Function
    def prune(self, ratio):
        weight_abs = self.weight.data.abs().view(-1)

        k = int(ratio * weight_abs.numel())
        if k == 0:
            return

        threshold = torch.topk(weight_abs, k, largest=False).values.max()

        mask = (self.weight.data.abs() > threshold).float()
        self.mask.data = mask



    # Quantization Function
    def quantize(self, num_bits):
        levels = 2 ** num_bits

        weight = self.weight.data
        w_min = weight.min()
        w_max = weight.max()

        if w_max == w_min:
            return

        scale = (w_max - w_min) / (levels - 1)

        q_weight = torch.round((weight - w_min) / scale)
        dequant_weight = q_weight * scale + w_min

        self.weight.data = dequant_weight