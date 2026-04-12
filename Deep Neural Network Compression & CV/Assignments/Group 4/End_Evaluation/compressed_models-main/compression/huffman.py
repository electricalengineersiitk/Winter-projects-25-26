"""
Huffman Coding for Neural Network Weight Compression.

Implements Huffman coding to encode quantized weight cluster indices.
Target: Reduce average bits per weight from 5 bits to ~3.57 bits.
"""

import heapq
from collections import Counter
import numpy as np
import torch
import torch.nn as nn

class HuffmanNode:
    """Node for Huffman tree"""
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq
    
    def is_leaf(self):
        return self.left is None and self.right is None

def build_huffman_tree(frequencies):
    """
    Build Huffman tree from symbol frequencies.
    
    Args:
        frequencies: Dict mapping symbols to their frequencies
        
    Returns:
        root: Root node of Huffman tree
    """
    if not frequencies:
        return None
    
    # Create priority queue of leaf nodes
    heap = []
    for symbol, freq in frequencies.items():
        heapq.heappush(heap, HuffmanNode(symbol=symbol, freq=freq))
    
    # Handle single symbol case
    if len(heap) == 1:
        node = heapq.heappop(heap)
        return HuffmanNode(freq=node.freq, left=node)
    
    # Build tree by combining lowest frequency nodes
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, parent)
    
    return heapq.heappop(heap)

def generate_codes(root, current_code="", codes=None):
    """
    Generate Huffman codes from tree.
    
    Args:
        root: Root node of Huffman tree
        current_code: Current code being built
        codes: Dict to store codes
        
    Returns:
        codes: Dict mapping symbols to their Huffman codes
    """
    if codes is None:
        codes = {}
    
    if root is None:
        return codes
    
    if root.is_leaf():
        # Use '0' for single symbol tree
        codes[root.symbol] = current_code if current_code else '0'
        return codes
    
    generate_codes(root.left, current_code + '0', codes)
    generate_codes(root.right, current_code + '1', codes)
    
    return codes

def huffman_encode(data):
    """
    Encode data using Huffman coding.
    
    Args:
        data: Array of symbols to encode
        
    Returns:
        encoded: Encoded bit string
        codebook: Dict mapping symbols to codes
        tree: Huffman tree root (for decoding)
    """
    # Count frequencies
    frequencies = Counter(data.flatten())
    
    # Build tree and generate codes
    tree = build_huffman_tree(dict(frequencies))
    codebook = generate_codes(tree)
    
    # Encode data
    encoded = ''.join(codebook[symbol] for symbol in data.flatten())
    
    return encoded, codebook, tree

def huffman_decode(encoded, codebook, shape):
    """
    Decode Huffman encoded data.
    
    Args:
        encoded: Encoded bit string
        codebook: Dict mapping symbols to codes
        shape: Original array shape
        
    Returns:
        decoded: Decoded array
    """
    # Reverse codebook
    reverse_codebook = {code: symbol for symbol, code in codebook.items()}
    
    decoded = []
    current_code = ""
    
    for bit in encoded:
        current_code += bit
        if current_code in reverse_codebook:
            decoded.append(reverse_codebook[current_code])
            current_code = ""
    
    return np.array(decoded).reshape(shape)

def calculate_average_bits(codebook, frequencies):
    """
    Calculate average bits per symbol.
    
    Args:
        codebook: Dict mapping symbols to codes
        frequencies: Dict mapping symbols to frequencies
        
    Returns:
        avg_bits: Average bits per symbol
    """
    total_bits = 0
    total_symbols = sum(frequencies.values())
    
    for symbol, freq in frequencies.items():
        if symbol in codebook:
            total_bits += freq * len(codebook[symbol])
    
    return total_bits / total_symbols if total_symbols > 0 else 0

def encode_layer_indices(module):
    """
    Apply Huffman encoding to a layer's cluster indices.
    
    Args:
        module: Layer with cluster_indices attribute
        
    Returns:
        encoded_data: Dict with encoding info
    """
    if not hasattr(module, 'cluster_indices') or module.cluster_indices is None:
        return None
    
    indices = module.cluster_indices.cpu().numpy()
    
    # Get mask to only encode non-zero positions
    if hasattr(module, 'mask'):
        mask = module.mask.cpu().numpy()
        # Only encode indices at non-zero mask positions
        nonzero_indices = indices[mask != 0]
    else:
        nonzero_indices = indices.flatten()
    
    if len(nonzero_indices) == 0:
        return None
    
    # Apply Huffman encoding
    encoded, codebook, tree = huffman_encode(nonzero_indices)
    
    # Calculate statistics
    frequencies = Counter(nonzero_indices.flatten())
    avg_bits = calculate_average_bits(codebook, frequencies)
    
    # Store on module
    module.huffman_encoded = encoded
    module.huffman_codebook = codebook
    
    return {
        'encoded_bits': len(encoded),
        'original_bits': len(nonzero_indices) * 32,  # Original float32
        'avg_bits_per_weight': avg_bits,
        'num_symbols': len(codebook),
        'compression_ratio': (len(nonzero_indices) * 32) / len(encoded) if len(encoded) > 0 else 0
    }

def huffman_encode_model(model):
    """
    Apply Huffman encoding to all quantized layers in the model.
    
    Args:
        model: PyTorch model with quantized layers
        
    Returns:
        stats: Dict with encoding statistics
    """
    stats = {
        'layers': {},
        'total_original_bits': 0,
        'total_encoded_bits': 0
    }
    
    for name, module in model.named_modules():
        if hasattr(module, 'cluster_indices'):
            layer_stats = encode_layer_indices(module)
            if layer_stats:
                stats['layers'][name] = layer_stats
                stats['total_original_bits'] += layer_stats['original_bits']
                stats['total_encoded_bits'] += layer_stats['encoded_bits']
                print(f"Huffman encoded {name}: {layer_stats['avg_bits_per_weight']:.2f} bits/weight")
    
    if stats['total_encoded_bits'] > 0:
        overall_ratio = stats['total_original_bits'] / stats['total_encoded_bits']
        print(f"\nHuffman encoding complete: {overall_ratio:.2f}x compression on indices")
    
    return stats

def get_compressed_size_bits(model):
    """
    Calculate total compressed size in bits.
    
    Accounts for:
    - Huffman encoded indices
    - Cluster centers (32-bit floats)
    - Sparse mask overhead
    """
    total_bits = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'huffman_encoded'):
            # Encoded indices
            total_bits += len(module.huffman_encoded)
            
            # Cluster centers (32 bits each)
            if hasattr(module, 'cluster_centers'):
                total_bits += len(module.cluster_centers) * 32
            
            # Codebook overhead (simplified)
            if hasattr(module, 'huffman_codebook'):
                total_bits += len(module.huffman_codebook) * 16  # Approximate
        
        # Mask overhead (1 bit per weight position for sparse storage)
        if hasattr(module, 'mask'):
            nonzero = (module.mask != 0).sum().item()
            total = module.mask.numel()
            # CSR format overhead
            total_bits += nonzero * 32  # Column indices
            total_bits += (module.mask.shape[0] + 1) * 32  # Row pointers
    
    return total_bits

def print_compression_summary(model, original_size_mb=228):
    """Print comprehensive compression summary"""
    total_bits = get_compressed_size_bits(model)
    compressed_size_mb = total_bits / (8 * 1024 * 1024)
    
    print("\n" + "="*60)
    print("COMPRESSION SUMMARY")
    print("="*60)
    print(f"Original model size: {original_size_mb:.2f} MB")
    print(f"Compressed size: {compressed_size_mb:.2f} MB")
    print(f"Compression ratio: {original_size_mb/compressed_size_mb:.2f}x")
    print("="*60)
    
    return compressed_size_mb
