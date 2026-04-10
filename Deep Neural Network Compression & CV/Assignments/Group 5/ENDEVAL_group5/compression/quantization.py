# compression/quantization.py — K-Means Weight Quantization

import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


def _kmeans_quantize_tensor(weight_np, n_clusters, layer_name=''):
    """
    Clusters the non-zero weights of a single tensor into k centroids.
    Pruned zeros are excluded from clustering and kept as zero.

    Returns a codebook dict:
        centroids   : float32 array of shape (k,)
        indices     : uint8 array — cluster assignment per non-zero weight
        nonzero_idx : int32 array — positions of non-zero weights in flat tensor
        shape       : original tensor shape
        n_clusters  : actual k used
        zero_count  : number of zero weights (pruned)
    """
    original_shape = weight_np.shape
    flat           = weight_np.flatten().astype(np.float32)
    nonzero_idx    = np.where(flat != 0.0)[0]
    zero_count     = len(flat) - len(nonzero_idx)

    if len(nonzero_idx) == 0:
        return {
            'centroids':   np.array([], dtype=np.float32),
            'indices':     np.array([], dtype=np.uint8),
            'nonzero_idx': nonzero_idx,
            'shape':       list(original_shape),
            'n_clusters':  0,
            'zero_count':  zero_count,
        }

    nonzero_vals   = flat[nonzero_idx].reshape(-1, 1)
    k              = min(n_clusters, len(np.unique(nonzero_vals)))
    w_min, w_max   = nonzero_vals.min(), nonzero_vals.max()
    init_centroids = np.linspace(w_min, w_max, k).reshape(-1, 1)

    km = KMeans(n_clusters=k, init=init_centroids,
                n_init=1, max_iter=300, random_state=42)
    km.fit(nonzero_vals)

    return {
        'centroids':   km.cluster_centers_.flatten().astype(np.float32),
        'indices':     km.labels_.astype(np.uint8),
        'nonzero_idx': nonzero_idx.astype(np.int32),
        'shape':       list(original_shape),
        'n_clusters':  k,
        'zero_count':  int(zero_count),
    }


def quantize_model(model, n_clusters_conv=256, n_clusters_fc=32):
    """
    Applies K-Means quantization to all Conv2d and Linear layers.

    Works on a deep copy — original model is not modified.

    Args:
        model          : pruned nn.Module (pruning must be made permanent first)
        n_clusters_conv: number of K-Means clusters for Conv2d layers
        n_clusters_fc  : number of K-Means clusters for Linear layers

    Returns:
        q_model   : quantized model (weights replaced with centroid values)
        codebooks : dict mapping layer name → codebook dict
    """
    model = copy.deepcopy(model)
    codebooks = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            k  = n_clusters_conv if isinstance(module, nn.Conv2d) else n_clusters_fc
            w  = module.weight.data.cpu().numpy()
            cb = _kmeans_quantize_tensor(w, k, name)
            codebooks[name] = cb

            if cb['n_clusters'] > 0:
                flat_r = np.zeros_like(w.flatten())
                flat_r[cb['nonzero_idx']] = cb['centroids'][cb['indices']]
                with torch.no_grad():
                    module.weight.data.copy_(
                        torch.from_numpy(flat_r.reshape(w.shape))
                        .to(module.weight.device)
                    )

    return model, codebooks


def get_quant_stats(codebooks, original_bits=32):
    """
    Computes theoretical compression metrics from the codebooks.

    Returns a stats dict including:
        overall_ratio    : theoretical compression ratio vs float32 baseline
        compressed_kb    : theoretical size of codebook representation
        avg_bits_weight  : average bits used per non-zero weight
    """
    total_w = total_nz = total_orig = total_quant = 0
    per_layer = []

    for name, cb in codebooks.items():
        n          = int(np.prod(cb['shape']))
        nz         = len(cb['nonzero_idx'])
        k          = cb['n_clusters']
        bits_orig  = n * original_bits
        idx_bits   = int(np.ceil(np.log2(k + 1e-9))) if k > 0 else 0
        bits_quant = nz * idx_bits + k * original_bits if k > 0 else 0

        per_layer.append({
            'name':     name,
            'shape':    cb['shape'],
            'k':        k,
            'sparsity': cb['zero_count'] / max(n, 1),
            'ratio':    bits_orig / max(bits_quant, 1),
        })
        total_w    += n
        total_nz   += nz
        total_orig += bits_orig
        total_quant += bits_quant

    return {
        'per_layer':        per_layer,
        'total_weights':    total_w,
        'total_nonzero':    total_nz,
        'overall_sparsity': (total_w - total_nz) / max(total_w, 1),
        'overall_ratio':    total_orig / max(total_quant, 1),
        'avg_bits_weight':  total_quant / max(total_nz, 1),
        'original_kb':      total_orig / 8 / 1024,
        'compressed_kb':    total_quant / 8 / 1024,
    }
