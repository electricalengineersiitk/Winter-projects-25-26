import torch
import heapq
from collections import Counter
from typing import Optional
import numpy as np

class HuffmanNode:
   
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        
       
        self.left: Optional['HuffmanNode'] = None
        self.right: Optional['HuffmanNode'] = None

    def __lt__(self, other):
        
        return self.freq < other.freq

def apply_huffman_encoding(model):
   
    print("[*] Applying Huffman Encoding")
    all_weights = []
    
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weights = module.weight.data.cpu().numpy().flatten()
            non_zero_weights = weights[weights != 0]
            all_weights.extend(non_zero_weights)
            
    if len(all_weights) == 0:
        return 0.0
        
   
    all_weights = np.round(all_weights, decimals=6)
    frequencies = Counter(all_weights)
    
   
    heap = [HuffmanNode(val, freq) for val, freq in frequencies.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
        
    
    codes = {}
    def generate_codes(node, current_code):
        if node is None:
            return
        if node.value is not None:
            codes[node.value] = current_code
            return
        generate_codes(node.left, current_code + "0")
        generate_codes(node.right, current_code + "1")
        
    root = heap[0] if heap else None
    generate_codes(root, "")
    
   
    total_bits = 0
    for val, freq in frequencies.items():
        total_bits += freq * len(codes[val])
        
   
    size_mb = total_bits / (8 * 1024 * 1024)
    
    
    print("\n   HUFFMAN DICTIONARY (Snippet)")
    print(f"   [-] Unique Weight Clusters: {len(frequencies)}")
    
   
    sorted_freqs = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    for val, freq in sorted_freqs[:5]: 
        code = codes[val]
        print(f"   [-] Value: {val:8.4f} | Freq: {freq:7,} | Huffman Code: {code} ({len(code)} bits)")
    if len(codes) > 5:
        print("   [-] ... (and so on)")
    print("   ------------------------------------\n")
    
    return size_mb