import torch
import numpy as np
from sklearn.cluster import KMeans

def apply_weight_sharing(model, bits: int = 4):
    
    n_clusters = 2 ** bits
    print(f"[*] Applying {bits}-bit Quantization (K-Means Weight Sharing)...")
    print(f"    Targeting {n_clusters} unique weight values per layer.")

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            
            weights = module.weight.data.cpu().numpy()
            
            
            non_zero_mask = weights != 0
            non_zero_weights = weights[non_zero_mask]

            if len(non_zero_weights) == 0:
                continue 
            
           
            non_zero_weights = non_zero_weights.reshape(-1, 1)

            #  Apply KMeans Clustering
           
            actual_clusters = min(n_clusters, len(non_zero_weights))
            kmeans = KMeans(n_clusters=actual_clusters, n_init=20, random_state=42)
            kmeans.fit(non_zero_weights)

            centroids = kmeans.cluster_centers_.flatten()
            labels = kmeans.labels_
            quantized_non_zeros = centroids[labels]

            quantized_weights = np.zeros_like(weights)
            quantized_weights[non_zero_mask] = quantized_non_zeros

           
            module.weight.data = torch.from_numpy(quantized_weights).to(module.weight.device)
            
    return model

def print_quantization_stats(model):
    """
    Prints how many unique weight values exist in each Linear layer.
    """
    print("--- QUANTIZATION STATS ---")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # count unique values, ignoring the exact 0.0 (which is our pruning mask)
            unique_weights = torch.unique(module.weight.data)
            num_unique = len(unique_weights) - (1 if 0.0 in unique_weights else 0)
            print(f"Layer '{name}': {num_unique} unique active weights + 1 zero-mask")
    print("--------------------------\n")

