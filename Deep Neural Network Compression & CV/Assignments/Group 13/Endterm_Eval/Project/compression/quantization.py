import torch
import numpy as np
from sklearn.cluster import KMeans
from compression.conv2d import modified_conv2d
from compression.linear import modified_linear

def quantize_model(model, k=32):
    """
    Applies K-Means quantization and stores indices for trained quantization[cite: 16, 186].
    """
    print(f"Starting K-Means quantization with k={k}...")
    
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (modified_conv2d, modified_linear)):
                # 1. Prepare weights and mask
                weights = m.weight.data.cpu().numpy()
                mask = m.mask.cpu().numpy().astype(bool)
                target_weights = weights[mask].reshape(-1, 1)
                
                if len(target_weights) > k:
                    # 2. Linear Initialization: Spaced equally between min and max
                    # This ensures large weights are well-represented
                    min_w, max_w = target_weights.min(), target_weights.max()
                    initial_centroids = np.linspace(min_w, max_w, k).reshape(-1, 1)

                    # 3. Perform K-Means clustering
                    kmeans = KMeans(n_clusters=k, init=initial_centroids, n_init=1, random_state=42)
                    kmeans.fit(target_weights)
                    
                    # 4. Update weights with cluster centers (centroids)
                    new_weights = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                    weights[mask] = new_weights
                    m.weight.data.copy_(torch.from_numpy(weights).to(m.weight.device))
                    
                    # 5. Store indices for Trained Quantization fine-tuning
                    # Create a full-size index matrix (un-masked weights get index 0)
                    full_indices = np.zeros(weights.shape, dtype=np.int64)
                    full_indices[mask] = kmeans.labels_
                    m.indices.copy_(torch.from_numpy(full_indices).to(m.indices.device))
                    
                    # Mark the layer as ready for centroid fine-tuning
                    m.is_quantized = True
    
    print("Quantization complete and cluster indices saved.")