"""
Modified Conv2d layer with pruning and quantization support.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

class ModifiedConv2d(nn.Conv2d):
    """
    Conv2d layer with support for:
    - L1 magnitude pruning with masks
    - K-Means weight quantization
    - Multiple forward modes (normal, pruned, quantized)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias, padding_mode, device, dtype)
        self.mode = 'normal'  # 'normal', 'pruned', 'quantized'
        
        # Pruning attributes
        self.register_buffer('mask', torch.ones_like(self.weight))
        
        # Quantization attributes
        self.cluster_centers = None
        self.cluster_indices = None
        self.huffman_encoded = None
        self.huffman_codebook = None
    
    def prune(self, threshold):
        """
        Apply L1 magnitude-based pruning.
        Weights with absolute value below threshold are masked to zero.
        
        Args:
            threshold: Pruning threshold (weights below this are pruned)
        """
        with torch.no_grad():
            self.mask = (torch.abs(self.weight) >= threshold).float()
            self.weight.data *= self.mask
            self.mode = 'pruned'
    
    def prune_by_percentile(self, percentile):
        """
        Prune weights below a given percentile of absolute values.
        
        Args:
            percentile: Percentage of weights to prune (0-100)
        """
        with torch.no_grad():
            abs_weights = torch.abs(self.weight.data)
            threshold = torch.quantile(abs_weights.flatten(), percentile / 100.0)
            self.prune(threshold.item())
    
    def quantize(self, k):
        """
        Quantize weights using K-Means clustering.
        
        Args:
            k: Number of clusters (e.g., 32 for 5-bit quantization)
        """
        with torch.no_grad():
            weight_np = self.weight.data.cpu().numpy()
            original_shape = weight_np.shape
            
            # Get non-zero weights (if pruned)
            if self.mode == 'pruned':
                mask_np = self.mask.cpu().numpy()
                nonzero_mask = mask_np.flatten() != 0
                nonzero_weights = weight_np.flatten()[nonzero_mask]
            else:
                nonzero_weights = weight_np.flatten()
                nonzero_mask = np.ones(weight_np.size, dtype=bool)
            
            if len(nonzero_weights) < k:
                k = max(1, len(nonzero_weights))
            
            # Apply K-Means clustering
            if len(nonzero_weights) > 0:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(nonzero_weights.reshape(-1, 1))
                
                self.cluster_centers = kmeans.cluster_centers_.flatten()
                
                # Create full indices array
                all_indices = np.zeros(weight_np.size, dtype=np.int32)
                all_indices[nonzero_mask] = kmeans.labels_
                self.cluster_indices = torch.tensor(
                    all_indices.reshape(original_shape),
                    device=self.weight.device
                )
                
                # Reconstruct quantized weights
                quantized_flat = np.zeros(weight_np.size)
                quantized_flat[nonzero_mask] = self.cluster_centers[kmeans.labels_]
                self.weight.data = torch.tensor(
                    quantized_flat.reshape(original_shape),
                    dtype=self.weight.dtype,
                    device=self.weight.device
                )
            
            self.mode = 'quantized'
    
    def get_quantized_weight(self):
        """Reconstruct weights from cluster centers and indices"""
        if self.cluster_centers is not None and self.cluster_indices is not None:
            centers_tensor = torch.tensor(
                self.cluster_centers,
                dtype=self.weight.dtype,
                device=self.weight.device
            )
            return centers_tensor[self.cluster_indices]
        return self.weight
    
    def forward(self, input):
        """Forward pass with mode-dependent weight handling"""
        if self.mode == 'normal':
            weight = self.weight
        elif self.mode == 'pruned':
            weight = self.weight * self.mask
        elif self.mode == 'quantized':
            weight = self.get_quantized_weight()
            if hasattr(self, 'mask'):
                weight = weight * self.mask
        else:
            weight = self.weight
        
        return F.conv2d(input, weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
    
    def get_sparsity(self):
        """Calculate sparsity of this layer"""
        total = self.weight.numel()
        zeros = (self.mask == 0).sum().item()
        return 100.0 * zeros / total
    
    def get_compression_stats(self):
        """Get compression statistics for this layer"""
        stats = {
            'total_weights': self.weight.numel(),
            'nonzero_weights': (self.mask != 0).sum().item(),
            'sparsity': self.get_sparsity(),
            'mode': self.mode
        }
        if self.cluster_centers is not None:
            stats['num_clusters'] = len(self.cluster_centers)
        return stats


# Backward compatibility alias
modified_conv2d = ModifiedConv2d
