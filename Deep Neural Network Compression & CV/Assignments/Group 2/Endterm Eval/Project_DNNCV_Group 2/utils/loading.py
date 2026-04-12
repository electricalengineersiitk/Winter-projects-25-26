
import numpy as np
import torch
import heapq
from collections import Counter

def huffman_encode(data):
    data_flat = data.flatten()
    freq = Counter(data_flat)
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]: pair[1] = '0' + pair[1]
        for pair in hi[1:]: pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    codebook = dict(heap[0][1:])
    bitstream = "".join([codebook[val] for val in data_flat])
    return bitstream, codebook

def save_model_npz(model, path):
    compressed_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'hashtable') and module.hashtable is not None:
            bits, book = huffman_encode(module.assignments.cpu().numpy())
            compressed_dict[f"{name}_bits"] = bits
            compressed_dict[f"{name}_book"] = book
            compressed_dict[f"{name}_shape"] = module.assignments.shape
            compressed_dict[f"{name}_centroids"] = module.hashtable.cpu().detach().numpy()
        elif hasattr(module, 'weight') and module.weight is not None:
            compressed_dict[f"{name}_weight"] = module.weight.data.cpu().numpy()
            if hasattr(module, 'bias') and module.bias is not None:
                compressed_dict[f"{name}_bias"] = module.bias.data.cpu().numpy()
    np.savez_compressed(path, **compressed_dict)
    print(f"Model saved to {path}")

def load_model_from_npz(model, path, device):
    data = np.load(path, allow_pickle=True)
    for name, module in model.named_modules():
        if f"{name}_bits" in data:
            bits, book, shape, centroids = str(data[f"{name}_bits"]), data[f"{name}_book"].item(), data[f"{name}_shape"], data[f"{name}_centroids"]
            rev_book = {v: k for k, v in book.items()}
            decoded, buffer = [], ""
            for bit in bits:
                buffer += bit
                if buffer in rev_book:
                    decoded.append(rev_book[buffer])
                    buffer = ""
            weights = centroids[np.array(decoded)].reshape(shape)
            module.weight.data = torch.from_numpy(weights).to(device)
        elif f"{name}_weight" in data:
            module.weight.data = torch.from_numpy(data[f"{name}_weight"]).to(device)
            if f"{name}_bias" in data:
                module.bias.data = torch.from_numpy(data[f"{name}_bias"]).to(device)
    return model