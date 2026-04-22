import torch
from torch import nn
from sklearn.cluster import KMeans
import numpy as np

class modified_conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', 
                 device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, 
                         padding, dilation, groups, bias, padding_mode, device, dtype)
        self.mode = 'normal'
        
        self.register_buffer('mask', torch.ones_like(self.weight))
        self.register_buffer('cluster_map', torch.zeros_like(self.weight, dtype=torch.long))
        self.cluster_centers = nn.Parameter(torch.empty(0, dtype=torch.float32))

    def prune(self, threshold):
        self.mask = (torch.abs(self.weight.data) >= threshold).float()
        self.weight.data.mul_(self.mask)
        self.mode = 'prune'

    def quantize(self, k):
        weight_np = self.weight.data.cpu().numpy()
        non_zero_mask = weight_np != 0
        non_zero_weights = weight_np[non_zero_mask].reshape(-1, 1)

        if len(non_zero_weights) >= k:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1)
            kmeans.fit(non_zero_weights)

            centers = kmeans.cluster_centers_.astype(np.float32).reshape(-1)
            centers = np.append(centers, np.float32(0.0)) 
            
            full_labels = np.full(weight_np.shape, k, dtype=np.int64)
            full_labels[non_zero_mask] = kmeans.labels_.astype(np.int64)

            self.cluster_centers.data = torch.from_numpy(centers).to(self.weight.device)
            self.cluster_map = torch.from_numpy(full_labels).to(self.weight.device)
            
            self.weight.requires_grad = False
            self.cluster_centers.requires_grad = True
            
        self.mode = 'quantize'

    def forward(self, input):
        if self.mode == 'normal':
            return self._conv_forward(input, self.weight, self.bias)
            
        elif self.mode == 'prune':
            masked_weight = self.weight * self.mask
            return self._conv_forward(input, masked_weight, self.bias)
            
        elif self.mode == 'quantize':
            quantized_weight = self.cluster_centers[self.cluster_map]
            return self._conv_forward(input, quantized_weight, self.bias)
            
        return self._conv_forward(input, self.weight, self.bias)