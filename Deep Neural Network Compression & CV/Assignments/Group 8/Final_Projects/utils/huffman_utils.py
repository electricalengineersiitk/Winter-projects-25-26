
def print_huffman_snippet(freq, codebook, top_k=5):
    print("\n   --- HUFFMAN DICTIONARY (Snippet) ---")

    sorted_items = sorted(freq.items(), key=lambda x: -x[1])

    print(f"   [-] Unique Weight Clusters: {len(freq)}")

    for val, f in sorted_items[:top_k]:
        code = codebook[val]
        print(f"   [-] Value: {val:>8.4f} | Freq: {f:>7} | Huffman Code: {code} ({len(code)} bits)")

    print("   [-] ... (and so on)")
    print("   ------------------------------------\n")


from collections import Counter

def apply_huffman_to_npz(npz_path):
    import numpy as np
    data = np.load(npz_path)

    total_bits = 0
    total_original_bits = 0

    combined = []

    for key in data:
        arr = data[key]
        combined.extend(arr.flatten().tolist())

    freq = Counter(combined)

    from compression.huffman import build_huffman_tree
    codebook = build_huffman_tree(freq)

    encoded_bits = sum(len(codebook[val]) * count for val, count in freq.items())
    total_bits = encoded_bits
    total_original_bits = len(combined) * 32

    print_huffman_snippet(freq, codebook)

    compressed_mb = total_bits / (8 * 1024 * 1024)
    original_mb = total_original_bits / (8 * 1024 * 1024)

    print(f"   [-] Huffman Encoded Size: {compressed_mb:.4f} MB")
    print(f"   [-] Compression Ratio: {original_mb/compressed_mb:.2f}x")