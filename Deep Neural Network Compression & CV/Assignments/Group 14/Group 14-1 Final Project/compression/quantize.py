import numpy as np
from sklearn.cluster import KMeans

def quantize_tensor(tensor, mask, bits=4):
    num_clusters = 2 ** bits
    weights = tensor.cpu().detach().numpy()
    mask_np = mask.cpu().detach().numpy()

    active_indices = np.where(mask_np == 1)
    active_weights = weights[active_indices].reshape(-1, 1)

    if len(active_weights) > num_clusters:
        kmeans = KMeans(n_clusters=num_clusters, n_init=1, random_state=42)
        kmeans.fit(active_weights)

        quantized_active = kmeans.cluster_centers_[kmeans.labels_].flatten()
        weights[active_indices] = quantized_active

    return weights


def quantize_model(model, masks, bits=4):
    quantized_state_dict = {}

    for name, param in model.named_parameters():
        if 'weight' in name and name in masks:
            print(f"Quantizing {name}...")
            quantized_weights = quantize_tensor(param.data, masks[name], bits)
            quantized_state_dict[name] = quantized_weights
        else:
            quantized_state_dict[name] = param.data.cpu().detach().numpy()

    return quantized_state_dict