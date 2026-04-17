# compression/huffman.py — Huffman Coding for quantized index streams

import heapq
import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


# ── Tree node ─────────────────────────────────────────────────────────────────

@dataclass(order=True)
class _HNode:
    freq:   int
    symbol: Any = field(compare=False, default=None)
    left:   Any = field(compare=False, default=None)
    right:  Any = field(compare=False, default=None)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_tree(symbols):
    freq = Counter(symbols.tolist())
    heap = [_HNode(f, s) for s, f in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        node = heapq.heappop(heap)
        return _HNode(node.freq, left=node, right=_HNode(0))
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, _HNode(a.freq + b.freq, left=a, right=b))
    return heap[0]


def _build_codebook(root):
    codes = {}
    def walk(node, prefix=''):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = prefix or '0'
            return
        walk(node.left,  prefix + '0')
        walk(node.right, prefix + '1')
    walk(root)
    return codes


def _bits_to_bytes(bitstring):
    pad    = (8 - len(bitstring) % 8) % 8
    padded = bitstring + '0' * pad
    out    = bytearray(int(padded[i:i+8], 2) for i in range(0, len(padded), 8))
    return bytes(out), pad


def bytes_to_bits(data, pad):
    bits = ''.join(f'{b:08b}' for b in data)
    return bits[:len(bits) - pad] if pad else bits


def decode_from_codebook(encoded_bytes, pad, huff_codebook, n):
    """Decode n symbols using the reverse of huff_codebook (no tree needed)."""
    reverse_cb = {v: k for k, v in huff_codebook.items()}
    bits = bytes_to_bits(encoded_bytes, pad)
    out, buf = [], ''
    for bit in bits:
        buf += bit
        if buf in reverse_cb:
            out.append(reverse_cb[buf])
            buf = ''
            if len(out) == n:
                break
    return np.array(out, dtype=np.uint8)


# ── Public API ────────────────────────────────────────────────────────────────

def huffman_encode_codebooks(codebooks):
    """
    Huffman-encodes the quantization index arrays for all layers.

    Args:
        codebooks : dict from quantize_model()

    Returns:
        encoded_data : serializable dict (no Python tree objects)
        huff_stats   : compression stats dict
    """
    encoded_data = {}
    per_layer    = []
    tb_before = tb_after = tc = 0

    for name, cb in codebooks.items():
        indices = cb['indices']
        k       = cb['n_clusters']
        n       = len(indices)

        if n == 0:
            encoded_data[name] = {
                **{kk: cb[kk] for kk in
                   ('centroids', 'nonzero_idx', 'shape', 'n_clusters', 'zero_count')},
                'encoded_bytes': b'', 'pad': 0,
                'n_indices': 0, 'huff_codebook': {},
            }
            per_layer.append({
                'name': name, 'k': k, 'n_indices': 0,
                'idx_bits_fixed': 0, 'idx_bits_huffman': 0,
                'centroid_bits': k * 32, 'avg_code_length': 0,
                'layer_huff_ratio': 1,
            })
            tc += k * 32
            continue

        root    = _build_tree(indices)
        huff_cb = _build_codebook(root)
        bits    = ''.join(huff_cb[s] for s in indices.tolist())
        enc, pad = _bits_to_bytes(bits)

        idx_fixed = int(np.ceil(np.log2(k + 1e-9))) * n
        idx_huff  = len(enc) * 8
        c_bits    = k * 32

        tb_before += idx_fixed
        tb_after  += idx_huff
        tc        += c_bits

        # NOTE: huffman_tree is NOT stored — rebuild at decode via huff_codebook
        encoded_data[name] = {
            **{kk: cb[kk] for kk in
               ('centroids', 'nonzero_idx', 'shape', 'n_clusters', 'zero_count')},
            'encoded_bytes': enc,
            'pad':           pad,
            'n_indices':     n,
            'huff_codebook': huff_cb,
        }
        per_layer.append({
            'name':             name,
            'k':                k,
            'n_indices':        n,
            'idx_bits_fixed':   idx_fixed,
            'idx_bits_huffman': idx_huff,
            'centroid_bits':    c_bits,
            'avg_code_length':  round(idx_huff / n, 3),
            'layer_huff_ratio': round(idx_fixed / max(idx_huff, 1), 3),
        })

    huff_stats = {
        'per_layer':              per_layer,
        'total_bits_before_huff': tb_before + tc,
        'total_bits_after_huff':  tb_after  + tc,
        'huffman_ratio':          round((tb_before + tc) / max(tb_after + tc, 1), 4),
        'size_before_huff_kb':    round((tb_before + tc) / 8 / 1024, 2),
        'size_after_huff_kb':     round((tb_after  + tc) / 8 / 1024, 2),
    }
    return encoded_data, huff_stats


def huffman_decode_codebooks(encoded_data):
    """
    Decodes Huffman-encoded index streams back to quantization codebooks.

    Args:
        encoded_data : dict loaded from the .pkl archive

    Returns:
        codebooks : same format as quantize_model() output
    """
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
