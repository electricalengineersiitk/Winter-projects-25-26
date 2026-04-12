import numpy as np
import heapq
import os
from collections import Counter


class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq   = freq
        self.left   = None
        self.right  = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(frequencies):
    heap = [HuffmanNode(s, f) for s, f in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        l        = heapq.heappop(heap)
        r        = heapq.heappop(heap)
        merged   = HuffmanNode(None, l.freq + r.freq)
        merged.left  = l
        merged.right = r
        heapq.heappush(heap, merged)
    return heap[0]


def build_huffman_codes(node, prefix="", codes=None):
    if codes is None:
        codes = {}
    if node.symbol is not None:
        codes[node.symbol] = prefix or "0"
    else:
        build_huffman_codes(node.left,  prefix + "0", codes)
        build_huffman_codes(node.right, prefix + "1", codes)
    return codes


def huffman_encode(data):
    flat        = data.flatten().tolist()
    frequencies = Counter(flat)
    tree        = build_huffman_tree(frequencies)
    codes       = build_huffman_codes(tree)
    bitstring   = "".join(codes[s] for s in flat)
    pad_len     = (8 - len(bitstring) % 8) % 8
    padded      = bitstring + "0" * pad_len
    packed      = np.packbits(np.array(list(padded), dtype=np.uint8))
    return packed, pad_len, codes, len(flat)


def avg_bits_per_symbol(codes, frequencies):
    total = sum(frequencies.values())
    return sum(
        len(codes[s]) * (f / total)
        for s, f in frequencies.items()
    )


def huffman_compress_model(model, path):
    huff_arrays   = {}
    weight_arrays = {}

    total_bits_huffman = 0
    total_bits_raw     = 0

    for name, param in model.named_parameters():
        w     = param.data.cpu().numpy()
        w_int = (w * 1000).astype(np.int16)

        packed, pad_len, codes, orig_len = huffman_encode(w_int)

        # huffman.npz — bitstreams only
        huff_arrays[f"huff_data_{name}"] = packed
        huff_arrays[f"shape_{name}"]     = np.array(w.shape)
        huff_arrays[f"orig_len_{name}"]  = np.array(orig_len)
        huff_arrays[f"pad_len_{name}"]   = np.array(pad_len)
        huff_arrays[f"code_syms_{name}"] = np.array(
            list(codes.keys()), dtype=np.int32
        )
        huff_arrays[f"code_lens_{name}"] = np.array(
            [len(c) for c in codes.values()], dtype=np.int32
        )

        # huffman_weights.npz — float16 weights for loading
        weight_arrays[f"param_{name}"] = w.astype(np.float16)

        flat        = w_int.flatten().tolist()
        frequencies = Counter(flat)
        avg_bits    = avg_bits_per_symbol(codes, frequencies)
        total_bits_huffman += avg_bits * orig_len
        total_bits_raw     += 16 * orig_len

    # buffers go into both files
    for name, buf in model.named_buffers():
        huff_arrays[f"buffer_{name}"]   = buf.cpu().numpy()
        weight_arrays[f"buffer_{name}"] = buf.cpu().numpy()

    # summary stats — only in huffman.npz
    total_params = total_bits_raw / 16
    avg_bpw      = total_bits_huffman / max(total_params, 1)

    huff_arrays["huffman_total_bits"]  = np.array(total_bits_huffman)
    huff_arrays["raw_total_bits"]      = np.array(total_bits_raw)
    huff_arrays["huffman_ratio"]       = np.array(
        total_bits_raw / total_bits_huffman
        if total_bits_huffman > 0 else 0
    )
    huff_arrays["avg_bits_per_weight"] = np.array(avg_bpw)

    # save huffman.npz
    np.savez_compressed(path, **huff_arrays)
    huff_mb = os.path.getsize(path) / (1024 * 1024)

    # save huffman_weights.npz
    weights_path = path.replace(".npz", "_weights.npz")
    np.savez_compressed(weights_path, **weight_arrays)
    weights_mb = os.path.getsize(weights_path) / (1024 * 1024)

    print(f"[Huffman] Saved stats   → {path}          ({huff_mb:.2f} MB)")
    print(f"[Huffman] Saved weights → {weights_path}  ({weights_mb:.2f} MB)")
    print(f"[Huffman] Avg bits/weight  : {avg_bpw:.4f}")
    print(f"[Huffman] Gain over int16  : "
          f"{total_bits_raw / max(total_bits_huffman, 1):.2f}x")

    return huff_mb, weights_mb