import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

class modified_conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        
        self.register_buffer('mask', torch.ones_like(self.weight))
        self.mode = 'normal'
        self.hashtable = None
        self.assignments = None
        self._hook_registered = False

    def prune(self, threshold):
        self.mask.copy_((torch.abs(self.weight) >= threshold).float())
        self.weight.data.mul_(self.mask)
        self.mode = 'prune'

        if self.weight.requires_grad and not self._hook_registered:
            self.weight.register_hook(lambda grad: grad.mul_(self.mask))
            self._hook_registered = True

    def quantize(self, k):
        active_weights = self.weight.data[self.mask.bool()].cpu().numpy().reshape(-1, 1)

        if len(active_weights) == 0:
            return

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=1)
        kmeans.fit(active_weights)

        self.hashtable = nn.Parameter(torch.tensor(kmeans.cluster_centers_.flatten(), device=self.weight.device, dtype=self.weight.dtype))
        self.assignments = torch.zeros_like(self.weight, dtype=torch.long)
        
        labels = torch.tensor(kmeans.labels_, device=self.weight.device, dtype=torch.long)
        self.assignments[self.mask.bool()] = labels
        
        self.mode = 'quantize'
        print(f"Conv2d layer quantized to {k} clusters.")

    def forward(self, input):
        # we used the same flow as explained 
        if self.mode == 'normal':
            W = self.weight
        elif self.mode == 'prune':
            W = (self.weight * self.mask)
        elif self.mode == 'quantize':
            W = self.hashtable[self.assignments] * self.mask
            
        return F.conv2d(input, W, self.bias, self.stride, self.padding, self.dilation, self.groups)