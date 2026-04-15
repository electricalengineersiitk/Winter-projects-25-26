import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

class modified_linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.mode = 'normal'
        
        # MENTOR FIX 1: Named exactly 'mask'
        self.register_buffer('mask', torch.ones_like(self.weight))
        
        # MENTOR FIX 2: map must be long (int64) for PyTorch indexing
        self.register_buffer('cluster_map', torch.zeros_like(self.weight, dtype=torch.long))
        
        # MENTOR FIX 3: centers must be a Parameter to survive the 1 epoch QAT
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
            return F.linear(input, self.weight, self.bias)
            
        elif self.mode == 'prune':
            masked_weight = self.weight * self.mask
            return F.linear(input, masked_weight, self.bias)
            
        elif self.mode == 'quantize':
            quantized_weight = self.cluster_centers[self.cluster_map]
            return F.linear(input, quantized_weight, self.bias)
            
        return F.linear(input, self.weight, self.bias)