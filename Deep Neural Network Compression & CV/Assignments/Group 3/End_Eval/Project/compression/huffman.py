## huffman coding is the final step in compression stage of our project
## after pruning and quantization, the cluster indices are not evenly distributed
## so huffman coding makes use of this by giving shorter codes to frequently used indices and longer codes to less used ones so that more frequent codes will get shorter binary reperesentation and less frequents will get more longer ones
## this optimizes the space used to store all the indices

import numpy as np
from collections import Counter
import heapq

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


class HuffmanNode: # a single node in the huffman tree either a leaf or an internal node
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freq_dict): # builds the tree by repeatedly merging the two least frequent nodes
    heap = [HuffmanNode(symbol=s, freq=f) for s, f in freq_dict.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        node = heapq.heappop(heap)
        root = HuffmanNode(freq=node.freq, left=node)
        return root

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(
            freq=left.freq + right.freq,
            left=left,
            right=right
        )
        heapq.heappush(heap, merged)

    return heap[0]


def generate_huffman_codes(root, prefix="", code_map=None): # traverses the tree to assign binary codes left gets 0 and right gets 1
    if code_map is None:
        code_map = {}

    if root is None:
        return code_map

    if root.symbol is not None:
        code_map[root.symbol] = prefix if prefix else "0"
        return code_map

    generate_huffman_codes(root.left, prefix + "0", code_map)
    generate_huffman_codes(root.right, prefix + "1", code_map)

    return code_map


def compute_huffman_bits(data, code_map): # counts total bits needed to encode all the data using the huffman codes
    total_bits = sum(len(code_map[s]) for s in data)
    avg_bits = total_bits / len(data) if len(data) > 0 else 0
    return total_bits, avg_bits


def huffman_encode_layer(codebook): # applies huffman coding to one layers cluster indices and then it calculates the savings
    labels = codebook['labels']
    k = codebook['num_clusters']
    fixed_bits = codebook['bits']

    # counts how many times each cluster index appears
    freq = Counter(labels)

    tree = build_huffman_tree(freq)
    codes = generate_huffman_codes(tree)

    n = len(labels)
    fixed_total_bits = n * fixed_bits                # this is size without huffman
    huffman_total_bits, avg_bits = compute_huffman_bits(labels, codes)
    codebook_bits = k * 32                           

    savings_pct = (1 - huffman_total_bits / fixed_total_bits) * 100 if fixed_total_bits > 0 else 0

    return {
        'codes': codes,
        'n_symbols': n,
        'fixed_bits_total': fixed_total_bits,
        'huffman_bits_total': huffman_total_bits,
        'avg_bits_per_weight': avg_bits,
        'codebook_bits': codebook_bits,
        'savings_pct': savings_pct,
    }


def huffman_encode_model(codebooks): # applies huffman coding across all quantized layers and reports total savings
    print(f"\n{'='*60}")
    print(f"  STAGE 3: HUFFMAN CODING (Lossless Compression)")
    print(f"{'='*60}")

    huffman_results = {}
    total_fixed_bits = 0
    total_huffman_bits = 0
    total_codebook_bits = 0

    for name, codebook in codebooks.items():
        result = huffman_encode_layer(codebook)
        huffman_results[name] = result

        total_fixed_bits += result['fixed_bits_total']
        total_huffman_bits += result['huffman_bits_total']
        total_codebook_bits += result['codebook_bits']

        print(f"  {name:40s}  "
              f"fixed: {result['fixed_bits_total']:10d} bits  "
              f"huffman: {result['huffman_bits_total']:10d} bits  "
              f"avg: {result['avg_bits_per_weight']:.2f} b/w  "
              f"saving: {result['savings_pct']:.1f}%")

    total_before = total_fixed_bits + total_codebook_bits
    total_after = total_huffman_bits + total_codebook_bits

    print(f"\n[Huffman] Total before Huffman: {total_before:,} bits  "
          f"({total_before/8/1024:.1f} KB)")
    print(f"[Huffman] Total after Huffman:  {total_after:,} bits  "
          f"({total_after/8/1024:.1f} KB)")

    if total_before > 0:
        savings = (1 - total_after / total_before) * 100
        print(f"[Huffman] Additional savings from Huffman: {savings:.1f}%")

    return huffman_results, total_after


def compute_huffman_savings(codebooks): # quick utility to get the final compressed size numbers
    _, total_compressed_bits = huffman_encode_model(codebooks)
    return {
        'total_compressed_bits': total_compressed_bits,
        'total_compressed_bytes': total_compressed_bits / 8,
        'total_compressed_kb': total_compressed_bits / 8 / 1024,
        'total_compressed_mb': total_compressed_bits / 8 / 1024 / 1024,
    }
