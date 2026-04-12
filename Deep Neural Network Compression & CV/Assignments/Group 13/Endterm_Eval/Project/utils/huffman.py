import heapq
from collections import Counter
import numpy as np

class HuffmanNode:
    def __init__(self, char, freq):
        self.char, self.freq = char, freq
        self.left = self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    frequencies = Counter(data)
    heap = [HuffmanNode(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        n1, n2 = heapq.heappop(heap), heapq.heappop(heap)
        merged = HuffmanNode(None, n1.freq + n2.freq)
        merged.left, merged.right = n1, n2
        heapq.heappush(heap, merged)
    return heap[0]

def generate_codes(node, current_code="", codebook=None):
    if codebook is None: codebook = {}
    if node:
        if node.char is not None: codebook[node.char] = current_code
        generate_codes(node.left, current_code + "0", codebook)
        generate_codes(node.right, current_code + "1", codebook)
    return codebook

def huffman_encode(data):
    """Encodes data based on probability distribution[cite: 236]."""
    if not data: return "", {}
    root = build_huffman_tree(data)
    codebook = generate_codes(root)
    return "".join([codebook[s] for s in data]), codebook

def get_relative_indices(mask):
    """Stores index difference instead of absolute position[cite: 145]."""
    indices = np.where(mask.flatten() == 1)[0]
    if len(indices) == 0: return []
    rel_indices = [indices[0]]
    for i in range(1, len(indices)):
        diff = indices[i] - indices[i-1]
        # Filler zeros prevent overflow if difference exceeds bits
        while diff > 31: 
            rel_indices.append(31)
            diff -= 31
        rel_indices.append(diff)
    return rel_indices