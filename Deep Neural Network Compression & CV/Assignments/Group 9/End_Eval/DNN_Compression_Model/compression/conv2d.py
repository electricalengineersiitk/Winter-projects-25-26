import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

class modified_conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.mode = 'normal'
        self.mask = None
        self.cluster_centers = None
        self.weight_indices = None

    def prune(self, sparsity_threshold):
        tensor_abs = torch.abs(self.weight.data)
        tau = torch.quantile(tensor_abs, sparsity_threshold)
        self.mask = (tensor_abs >= tau).float()
        self.weight.data.mul_(self.mask)

    def quantize(self, k):
        weights_np = self.weight.data.cpu().numpy()
        mask_np = self.mask.cpu().numpy() if self.mask is not None else np.ones_like(weights_np)
        non_zero_weights = weights_np[mask_np == 1].reshape(-1, 1)

        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1)
        kmeans.fit(non_zero_weights)

        
        self.cluster_centers = nn.Parameter(torch.tensor(kmeans.cluster_centers_, device=self.weight.device, dtype=torch.float32))
        
        full_indices = np.zeros_like(weights_np, dtype=np.uint8)
        full_indices[mask_np == 1] = kmeans.labels_
        self.weight_indices = torch.tensor(full_indices, device=self.weight.device)

    def forward(self, input):
        if self.mode == 'normal':
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.mode == 'prune':
            return F.conv2d(input, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.mode == 'quantize':
            reconstructed_weight = self.cluster_centers[self.weight_indices.long()]
            if len(reconstructed_weight.shape) > 4:
                reconstructed_weight = reconstructed_weight.squeeze(-1)
            if self.mask is not None:
                reconstructed_weight = reconstructed_weight * self.mask
            return F.conv2d(input, reconstructed_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
