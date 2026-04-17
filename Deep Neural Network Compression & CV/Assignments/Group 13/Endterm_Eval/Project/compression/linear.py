import torch
import torch.nn as nn

class modified_linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Buffer for pruning: 1 for active connections, 0 for pruned
        self.register_buffer('mask', torch.ones(self.weight.data.shape))
        
        # NEW: Buffer for quantization: Stores cluster index (0 to k-1) for each weight
        self.register_buffer('indices', torch.zeros(self.weight.data.shape, dtype=torch.long))
        
        # NEW: Flag to indicate if the layer is ready for centroid fine-tuning
        self.is_quantized = False

    def forward(self, x):
        # Ensure pruned weights stay zero
        if self.mask is not None:
            self.weight.data.mul_(self.mask)
        return super().forward(x)

    def prune(self, threshold):
        pass

    def quantize(self, k):
        pass