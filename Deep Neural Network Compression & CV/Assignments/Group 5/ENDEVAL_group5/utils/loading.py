# utils/loading.py — Model and compressed archive loading

import torch
import pickle
import numpy as np
import os

from models import miniVGG
from compression.huffman import decode_from_codebook, bytes_to_bits


def load_model(weights_path, device):
    """Load a trained miniVGG from a .pth state_dict file."""
    model = miniVGG().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f'Loaded weights from {weights_path}')
    return model


def _huffman_decode_codebooks(encoded_data):
    """Decode all layers from the serialized Huffman archive."""
    codebooks = {}
    for name, enc in encoded_data.items():
        n = enc['n_indices']
        if n == 0:
            indices = np.array([], dtype=np.uint8)
        else:
            indices = decode_from_codebook(
                enc['encoded_bytes'], enc['pad'], enc['huff_codebook'], n
            )
        codebooks[name] = {
            'centroids':   enc['centroids'],
            'indices':     indices,
            'nonzero_idx': enc['nonzero_idx'],
            'shape':       enc['shape'],
            'n_clusters':  enc['n_clusters'],
            'zero_count':  enc['zero_count'],
        }
    return codebooks


def load_compressed_model(archive_path, quantized_pth_path, device):
    """
    Reconstructs a runnable miniVGG from the 3-stage compressed archive.

    Steps:
        1. Load Huffman archive (.pkl)
        2. Huffman-decode the index streams
        3. Dequantize each layer (centroids[indices])
        4. Build a miniVGG shell and load the reconstructed weights
        5. Return eval-ready model

    Args:
        archive_path      : path to huffman_encoded.pkl
        quantized_pth_path: path to quantized_model.pth (for BN/bias params)
        device            : torch.device

    Returns:
        model : miniVGG in eval mode
    """
    print('[1/4] Loading archive...')
    with open(archive_path, 'rb') as f:
        archive = pickle.load(f)

    print('[2/4] Huffman-decoding index streams...')
    codebooks = _huffman_decode_codebooks(archive['encoded_data'])

    print('[3/4] Dequantizing weights...')
    weight_tensors = {}
    for name, cb in codebooks.items():
        n_total = int(np.prod(cb['shape']))
        flat    = np.zeros(n_total, dtype=np.float32)
        if cb['n_clusters'] > 0:
            flat[cb['nonzero_idx']] = cb['centroids'][cb['indices']]
        weight_tensors[name] = torch.from_numpy(flat.reshape(cb['shape']))

    print('[4/4] Rebuilding miniVGG...')
    model = miniVGG().to(device)

    if os.path.exists(quantized_pth_path):
        state = torch.load(quantized_pth_path, map_location=device)
        for name, tensor in weight_tensors.items():
            key = name + '.weight'
            if key in state:
                state[key] = tensor.to(device)
        model.load_state_dict(state)
        print(f'   BN/bias loaded from {quantized_pth_path}')
    else:
        print('   WARNING: quantized_model.pth not found — BN params will be random.')
        state = model.state_dict()
        for name, tensor in weight_tensors.items():
            key = name + '.weight'
            if key in state:
                state[key] = tensor
        model.load_state_dict(state)

    model.eval()
    print('Done. Model ready for inference.')
    return model
