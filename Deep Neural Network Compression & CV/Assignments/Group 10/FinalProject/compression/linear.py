import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
class modified_linear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.mode = 'normal'
        self.mask = None
        self.quantized_weight = None
    def prune(self,threshold):
        #this function should prune the weights which are below threshold
        self.mask = (torch.abs(self.weight.data) >= threshold).float()
        self.weight.data.mul_(self.mask)
        self.mode = 'prune'
        print(f"Pruned Linear layer with threshold {threshold}")
    def quantize(self,k):
        '''cluster the weights in k different cluster, and then use 2 things, 
        one is a cluster map which will be o(nxn) uint8 type and one array of clusters which wil
        be a O(k) sized fload 34 

        note that SK-learn is a cpu library!
        '''
        weight_cpu = self.weight.data.cpu().numpy()
        original_shape = weight_cpu.shape
        weight_flattened = weight_cpu.reshape(-1, 1)

        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(weight_flattened)

        cluster_map = kmeans.labels_.astype(np.uint8) 
        cluster_centers = kmeans.cluster_centers_.astype(np.float32)

        reconstructed_weights = cluster_centers[cluster_map].reshape(original_shape)
        
        self.quantized_weight = nn.Parameter(torch.from_numpy(reconstructed_weights).to(self.weight.device))
        
        self.mode = 'quantize'
        print(f"Quantized Linear layer to {k} clusters")
        pass
    def forward(self, input):
        if(self.mode == 'normal'):
            active_weight = self.weight
        elif(self.mode == 'prune'):
            active_weight = self.weight * self.mask.to(self.weight.device)
        elif(self.mode == 'quantize'):
            active_weight = self.quantized_weight
        return F.linear(input, active_weight, self.bias)

    
