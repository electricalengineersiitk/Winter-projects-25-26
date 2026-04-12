# huffman encoding for the quantized weight indices
# after quantization, some centroids are used way more often than others
# huffman assigns shorter binary codes to more frequent symbols
# this is lossless compression - we can perfectly reconstruct the original
# its the final stage of the deep compression pipeline

import numpy as np
import heapq
from collections import Counter,defaultdict


# huffman tree node
class HuffmanNode:
    def __init__(self,symbol=None,freq=0,left=None,right=None):
        self.symbol=symbol
        self.freq=freq
        self.left=left
        self.right=right

    def __lt__(self,other):
        return self.freq<other.freq # for the min-heap comparison

    def is_leaf(self):
        return self.left is None and self.right is None


def build_huffman_tree(freq_dict):
    # build the tree from symbol frequencies using a min-heap
    # start with leaf nodes, then keep merging the two smallest until one tree remains
    heap=[]
    for symbol,freq in freq_dict.items():
        node=HuffmanNode(symbol=symbol,freq=freq)
        heapq.heappush(heap,node)

    # edge case: only one unique symbol
    if len(heap)==1:
        node=heapq.heappop(heap)
        root=HuffmanNode(freq=node.freq,left=node)
        return root

    # merge nodes bottom-up (classic huffman algorithm)
    while len(heap)>1:
        left=heapq.heappop(heap)  # smallest freq
        right=heapq.heappop(heap) # second smallest
        parent=HuffmanNode(freq=left.freq+right.freq,
                           left=left,right=right)
        heapq.heappush(heap,parent)

    return heap[0] # root of the tree


def generate_codes(root):
    # traverse the tree with DFS to get the binary codes
    # left edge = 0, right edge = 1
    codes={}

    def dfs(node,code=""):
        if node is None:
            return
        if node.is_leaf():
            codes[node.symbol]=code if code else "0" # single symbol edge case
            return
        dfs(node.left,code+"0")
        dfs(node.right,code+"1")

    dfs(root)
    return codes


def huffman_encode(data_array):
    # encode an array of symbols (like cluster indices) using huffman coding
    # returns: encoded bitstring, the codebook, and frequency dict

    # step 1: count how often each symbol appears
    freq_dict=Counter(data_array.tolist())

    # step 2: build tree and get codes
    tree=build_huffman_tree(freq_dict)
    codebook=generate_codes(tree)

    # step 3: encode the data by replacing each symbol with its code
    encoded_bits=''.join(codebook[symbol] for symbol in data_array)

    return encoded_bits,codebook,freq_dict


def huffman_decode(encoded_bits,codebook):
    # decode a huffman-encoded bitstring back to the original symbols
    # just invert the codebook and match bit patterns
    inverse_codebook={code:symbol for symbol,code in codebook.items()}

    decoded=[]
    current_code=""
    for bit in encoded_bits:
        current_code+=bit
        if current_code in inverse_codebook:
            decoded.append(inverse_codebook[current_code])
            current_code=""

    return decoded


# compression metrics
def compute_compression_ratio(original_array,encoded_bits,
                              original_bits_per_element=32):
    # figure out how much compression we actually got
    # comparing original size (32 bits per float) vs huffman encoded size
    original_size_bits=len(original_array)*original_bits_per_element
    compressed_size_bits=len(encoded_bits)

    ratio=original_size_bits/compressed_size_bits if compressed_size_bits>0 else float('inf')

    avg_code_length=(compressed_size_bits/len(original_array)
                     if len(original_array)>0 else 0)

    stats={
        'original_size_bits':original_size_bits,
        'compressed_size_bits':compressed_size_bits,
        'original_size_bytes':original_size_bits//8,
        'compressed_size_bytes':compressed_size_bits//8,
        'compression_ratio':ratio,
        'avg_bits_per_symbol':avg_code_length,
        'num_symbols':len(original_array)
    }

    return ratio,stats


def compute_entropy(freq_dict):
    # shannon entropy of the symbol distribution
    # tells us the theoretical minimum bits per symbol
    # formula: H = -sum(p * log2(p)) which is same as BCE but for multiple classes
    total=sum(freq_dict.values())
    entropy=0.0
    for count in freq_dict.values():
        if count>0:
            p=count/total
            entropy-=p*np.log2(p) # same entropy formula from information theory
    return entropy
