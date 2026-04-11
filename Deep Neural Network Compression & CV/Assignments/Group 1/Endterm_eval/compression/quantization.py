# weight quantization using k-means clustering
# this is very similar to what we did in assignment 5 with kmeans_matrix()
# but instead of clustering random matrix elements, we cluster the actual neural network weights
#
# approach:
#   1. for each layer, collect the non-zero (surviving after pruning) weights
#   2. cluster them into k groups using k-means (k = 2^bits, so 16 for 4-bit)
#   3. replace each weight with its cluster centroid
#   4. store only the cluster index per weight + the centroid codebook
# this drastically reduces unique weight values which helps with huffman later

import torch
import numpy as np
from sklearn.cluster import KMeans
from .linear import CompressedLinear
from .conv2d import CompressedConv2d

def quantize_weights(weight_array,mask_array,num_clusters=16):
    # quantize non-zero weights using k-means (similar idea to assignment 5)
    # extract the non-zero weights first, then cluster them
    flat_weights=weight_array.flatten()
    flat_mask=mask_array.flatten()
    nonzero_indices=np.where(flat_mask>0)[0] # positions where mask is active
    nonzero_weights=flat_weights[nonzero_indices]

    if len(nonzero_weights)==0:
        # everything was pruned, nothing to quantize
        return weight_array,np.zeros_like(flat_weights,dtype=np.int32),\
               np.zeros(num_clusters)

    # making sure we dont ask for more clusters than we have unique values
    num_unique=len(np.unique(nonzero_weights))
    actual_clusters=min(num_clusters,num_unique)

    # k-means clustering on the weight values (reshaped to column vector just like in assignment 5)
    kmeans=KMeans(n_clusters=actual_clusters,n_init=10,
                  random_state=42,max_iter=300)
    kmeans.fit(nonzero_weights.reshape(-1,1)) # reshape to column like flat_M in assignment 5

    # centroids are the new weight values, labels tell us which cluster each weight belongs to
    centroids=kmeans.cluster_centers_.flatten()
    labels=kmeans.labels_

    # build the quantized weight array - replace each weight with its centroid
    quantized_flat=np.zeros_like(flat_weights)
    index_array=np.zeros(len(flat_weights),dtype=np.int32)

    for i,idx in enumerate(nonzero_indices):
        quantized_flat[idx]=centroids[labels[i]] # replace with centroid value
        index_array[idx]=labels[i]               # store cluster assignment

    quantized_weights=quantized_flat.reshape(weight_array.shape) # back to original shape

    return quantized_weights,index_array,centroids


def apply_quantization(model,num_clusters=16):
    # apply k-means quantization to all CompressedLinear layers
    # modifies weights in-place: each weight becomes its cluster centroid
    print(f"\n--- Applying Quantization ({num_clusters} clusters) ---")

    quantization_info={}

    for name,module in model.named_modules():
        if isinstance(module, (CompressedLinear, CompressedConv2d)):
            weight_np=module.weight.data.cpu().numpy()
            mask_np=module.mask.data.cpu().numpy()

            quantized,indices,centroids=quantize_weights(
                weight_np,mask_np,num_clusters)

            # update model weights with the quantized values
            module.weight.data=torch.tensor(
                quantized,dtype=torch.float32).to(module.weight.device)

            # zero out pruned weights again just to be safe (mask enforcement)
            module.weight.data*=module.mask

            # count how many unique values we ended up with
            active=quantized[mask_np>0]
            num_unique=len(np.unique(active)) if len(active)>0 else 0

            quantization_info[name]={
                'indices':indices,
                'centroids':centroids,
                'num_unique_weights':num_unique
            }

            print(f"  Layer '{name}':")
            print(f"    Unique weight values: {num_unique}")
            print(f"    Centroids: {centroids[:8]}{'...' if len(centroids)>8 else ''}")

    return quantization_info
