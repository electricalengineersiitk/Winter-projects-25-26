import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

class ModifiedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.mode = 'normal'
        self.register_buffer('mask', torch.ones_like(self.weight))
        
        self.cluster_centers = None
        self.cluster_indices = None
        self.huffman_encoded = None
        self.huffman_codebook = None
    
    def prune(self, threshold):
        with torch.no_grad():
            self.mask = (torch.abs(self.weight) >= threshold).float()
            self.weight.data *= self.mask
            self.mode = 'pruned'
            
    def quantize(self, k):
        with torch.no_grad():
            weight_np = self.weight.data.cpu().numpy()
            original_shape = weight_np.shape
            
            mask_np = self.mask.cpu().numpy()
            nonzero_mask = mask_np.flatten() != 0
            nonzero_weights = weight_np.flatten()[nonzero_mask]
            
            if len(nonzero_weights) < k:
                k = max(1, len(nonzero_weights))
                
            if len(nonzero_weights) > 0:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(nonzero_weights.reshape(-1, 1))
                
                self.cluster_centers = kmeans.cluster_centers_.flatten()
                all_indices = np.zeros(weight_np.size, dtype=np.int32)
                all_indices[nonzero_mask] = kmeans.labels_
                
                self.cluster_indices = torch.tensor(
                    all_indices.reshape(original_shape),
                    device=self.weight.device
                )
                
                quantized_flat = np.zeros(weight_np.size)
                quantized_flat[nonzero_mask] = self.cluster_centers[kmeans.labels_]
                self.weight.data = torch.tensor(
                    quantized_flat.reshape(original_shape),
                    dtype=self.weight.dtype,
                    device=self.weight.device
                )
            self.mode = 'quantized'

    def get_quantized_weight(self):
        if self.cluster_centers is not None and self.cluster_indices is not None:
            centers_tensor = torch.tensor(
                self.cluster_centers, dtype=self.weight.dtype, device=self.weight.device
            )
            return centers_tensor[self.cluster_indices]
        return self.weight
    
    def forward(self, input):
        if self.mode == 'normal':
            weight = self.weight
        elif self.mode == 'pruned':
            weight = self.weight * self.mask
        elif self.mode == 'quantized':
            weight = self.get_quantized_weight() * self.mask
        else:
            weight = self.weight
            
        return F.linear(input, weight, self.bias)