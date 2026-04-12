

from __future__ import annotations

import heapq
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from collections import Counter


@dataclass(order=True)
class _HuffmanNode:
    freq: int
    symbol: Optional[int] = field(default=None, compare=False)
    left:  Optional["_HuffmanNode"] = field(default=None, compare=False)
    right: Optional["_HuffmanNode"] = field(default=None, compare=False)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _build_tree(freq_table: Dict[int, int]) -> _HuffmanNode:
    heap = [_HuffmanNode(freq=f, symbol=s) for s, f in freq_table.items()]
    heapq.heapify(heap)

    # Edge case: single unique symbol
    if len(heap) == 1:
        only = heapq.heappop(heap)
        return _HuffmanNode(freq=only.freq, left=only)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        heapq.heappush(heap, _HuffmanNode(freq=lo.freq + hi.freq, left=lo, right=hi))

    return heap[0]


def _build_codes(
    node: Optional[_HuffmanNode],
    prefix: str,
    table: Dict[int, str],
) -> None:
    if node is None:
        return
    if node.is_leaf:
        table[node.symbol] = prefix or "0"
        return
    _build_codes(node.left,  prefix + "0", table)
    _build_codes(node.right, prefix + "1", table)



def _pack_bits(bitstring: str) -> Tuple[bytes, int]:
    padding = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * padding
    ba = bytearray(int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8))
    return bytes(ba), padding


def _unpack_bits(data: bytes, padding: int) -> str:
    bits = "".join(f"{b:08b}" for b in data)
    return bits[:-padding] if padding else bits

def _tensor_to_symbols(tensor: torch.Tensor) -> Tuple[List[int], str]:
    arr = tensor.cpu().numpy()
    dtype_str = str(arr.dtype)

    if np.issubdtype(arr.dtype, np.integer):
        symbols = arr.flatten().tolist()
    else:
        symbols = arr.astype(np.float16).view(np.uint16).flatten().tolist()

    return symbols, dtype_str


def _encode_tensor(tensor: torch.Tensor) -> dict:

    symbols, dtype_str = _tensor_to_symbols(tensor)
    freq_table = dict(Counter(symbols))

    root = _build_tree(freq_table)
    code_table: Dict[int, str] = {}
    _build_codes(root, "", code_table)

    bitstring = "".join(code_table[s] for s in symbols)
    packed, padding = _pack_bits(bitstring)

    return {
        "encoded":    np.frombuffer(packed, dtype=np.uint8),
        "padding":    np.int32(padding),
        "shape":      np.array(list(tensor.shape), dtype=np.int64),
        "dtype":      np.array([dtype_str], dtype=object),
        "codes_json": np.array([json.dumps(code_table)], dtype=object),
        "freq_json":  np.array([json.dumps(freq_table)], dtype=object),
        "n_symbols":  np.int64(len(symbols)),
    }


def _decode_tensor(arrays: dict) -> torch.Tensor:
    
    encoded:    np.ndarray = arrays["encoded"]
    padding:    int        = int(arrays["padding"])
    shape:      list       = arrays["shape"].tolist()
    dtype_str:  str        = str(arrays["dtype"][0])
    codes_json: str        = str(arrays["codes_json"][0])
    n_symbols:  int        = int(arrays["n_symbols"])

    code_table: Dict[str, int] = {v: int(k) for k, v in json.loads(codes_json).items()}

    bitstring = _unpack_bits(encoded.tobytes(), padding)

    # Decode symbols
    symbols: List[int] = []
    buf = ""
    for bit in bitstring:
        buf += bit
        if buf in code_table:
            symbols.append(code_table[buf])
            buf = ""
        if len(symbols) == n_symbols:
            break

    if len(symbols) != n_symbols:
        raise ValueError(
            f"Decode error: expected {n_symbols} symbols, got {len(symbols)}."
        )

    # Reconstruct numpy array
    if np.issubdtype(np.dtype(dtype_str), np.integer):
        arr = np.array(symbols, dtype=np.dtype(dtype_str)).reshape(shape)
    else:
        arr = (
            np.array(symbols, dtype=np.uint16)
            .view(np.float16)
            .astype(np.float32)
            .reshape(shape)
        )

    return torch.from_numpy(arr)


def save_model_npz(
    model: nn.Module,
    npz_path: str,
) -> None:
    
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    save_dict: Dict[str, np.ndarray] = {}

    print(f"[Huffman] Encoding {len(state_dict)} tensors …")
    original_bytes = 0
    for name, tensor in state_dict.items():
        original_bytes += tensor.numel() * tensor.element_size()
        encoded = _encode_tensor(tensor)
        safe_name = name.replace(".", "_")          
        for suffix, arr in encoded.items():
            save_dict[f"{safe_name}__{suffix}"] = arr

   
    key_map = {name.replace(".", "_"): name for name in state_dict}
    save_dict["__key_map__"] = np.array([json.dumps(key_map)], dtype=object)

    np.savez_compressed(str(npz_path.with_suffix("")), **save_dict)

   
    saved_as = npz_path.with_suffix(".npz")
    if saved_as != npz_path and saved_as.exists():
        saved_as.rename(npz_path)
        saved_as = npz_path

    compressed_bytes = saved_as.stat().st_size
    ratio = original_bytes / max(compressed_bytes, 1)
    print(
        f"[Huffman] Saved → {saved_as}\n"
        f"          Original : {original_bytes / 1024:.1f} KB\n"
        f"          Compressed: {compressed_bytes / 1024:.1f} KB\n"
        f"          Ratio     : {ratio:.2f}×  ({(1 - 1/ratio)*100:.1f}% saved)"
    )


def load_model_from_npz(
    model: nn.Module,
    npz_path: str,
    device: torch.device,
) -> nn.Module:

    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Compressed model not found: {npz_path}")

    print(f"[Huffman] Loading compressed model <- {npz_path}")
    data = np.load(str(npz_path), allow_pickle=True)

    key_map: Dict[str, str] = json.loads(str(data["__key_map__"][0]))

    layer_keys: Dict[str, Dict[str, np.ndarray]] = {}
    for npz_key in data.files:
        if npz_key == "__key_map__":
            continue
        if "__" not in npz_key:
            continue
        safe_name, suffix = npz_key.rsplit("__", 1)
        layer_keys.setdefault(safe_name, {})[suffix] = data[npz_key]

    decoded_sd: Dict[str, torch.Tensor] = {}
    for safe_name, arrays in layer_keys.items():
        original_name = key_map.get(safe_name, safe_name.replace("_", "."))
        tensor = _decode_tensor(arrays)
        decoded_sd[original_name] = tensor

    float_sd: Dict[str, torch.Tensor] = {}
    for name, tensor in decoded_sd.items():
        if torch.is_floating_point(tensor):
            float_sd[name] = tensor.float()
        else:
            float_sd[name] = tensor

    for key, tensor in float_sd.items():
        if not key.endswith("cluster_centers"):
            continue

        parts      = key.split(".")
        param_name = parts[-1]

        module = model
        try:
            for part in parts[:-1]:
                module = getattr(module, part)
        except AttributeError:
            print(f"  [WARN] Cannot navigate to {key} — skipping resize")
            continue

        setattr(
            module,
            param_name,
            nn.Parameter(
                torch.zeros(tensor.shape, dtype=torch.float32),
                requires_grad=True,
            ),
        )
        print(f"  Resized {key}: [0] -> {list(tensor.shape)}")

    missing, unexpected = model.load_state_dict(float_sd, strict=False)

    if missing:
        print(f"  [WARN] Missing  keys : {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected    : {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")

    for name, m in model.named_modules():
        if hasattr(m, "cluster_centers") and m.cluster_centers.numel() > 0:
            m.mode = "quantize"
        elif hasattr(m, "mask"):
            m.mode = "prune"

    for name, m in model.named_modules():
        if hasattr(m, "cluster_centers") and m.cluster_centers.numel() > 0:
            if m.cluster_centers.dtype != torch.float32:
                m.cluster_centers.data = m.cluster_centers.data.float()
                print(f"  dtype fixed: {name}.cluster_centers -> float32")

    model.to(device)
    model.eval()
    print(f"[Huffman] Model reconstructed and moved to {device}.")
    return model


def save_model_bin(model: nn.Module, bin_path: str) -> None:
   
    bin_path = Path(bin_path)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
   
    raw_sd = dict(model.state_dict())  

    for mod_name, module in model.named_modules():
        if getattr(module, "mode", None) == "quantize":
            if hasattr(module, "cluster_centers") and module.cluster_centers.numel() > 0:
                
                q_weight = module.cluster_centers[module.cluster_map].detach().cpu()
                
                if hasattr(module, "mask"):
                    q_weight = q_weight * module.mask.cpu()
                w_key = f"{mod_name}.weight" if mod_name else "weight"
                if w_key in raw_sd:
                    raw_sd[w_key] = q_weight

    state_dict = {k: v.cpu() for k, v in raw_sd.items()}
    key_map = {k: k for k in state_dict}

    with open(bin_path, "wb") as f:
        f.write(b"HCNN")
        f.write(struct.pack(">I", len(state_dict)))
        f.write(struct.pack(">I", len(json.dumps(key_map).encode())))
        f.write(json.dumps(key_map).encode())

        for name, tensor in state_dict.items():
            enc = _encode_tensor(tensor)
            meta = {
                "shape":      enc["shape"].tolist(),
                "dtype":      str(enc["dtype"][0]),
                "padding":    int(enc["padding"]),
                "n_symbols":  int(enc["n_symbols"]),
                "codes_json": str(enc["codes_json"][0]),
                "freq_json":  str(enc["freq_json"][0]),
            }
            name_b  = name.encode("utf-8")
            meta_b  = json.dumps(meta).encode("utf-8")
            data_b  = enc["encoded"].tobytes()

            f.write(struct.pack(">I", len(name_b)));  f.write(name_b)
            f.write(struct.pack(">I", len(meta_b)));  f.write(meta_b)
            f.write(struct.pack(">I", len(data_b)));  f.write(data_b)

    print(f"[Huffman/bin] Saved → {bin_path}  ({bin_path.stat().st_size / 1024:.1f} KB)")


def save_model_hex(model: nn.Module, hex_path: str) -> None:

    hex_path = Path(hex_path)
    hex_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    with open(hex_path, "w", encoding="utf-8") as f:
        f.write("HCNN_HEX\n")
        for name, tensor in state_dict.items():
            enc = _encode_tensor(tensor)
            meta = {
                "shape":      enc["shape"].tolist(),
                "dtype":      str(enc["dtype"][0]),
                "padding":    int(enc["padding"]),
                "n_symbols":  int(enc["n_symbols"]),
                "codes_json": str(enc["codes_json"][0]),
                "freq_json":  str(enc["freq_json"][0]),
            }
            f.write(f"LAYER {name}\n")
            f.write(f"META  {json.dumps(meta)}\n")
            f.write(f"DATA  {enc['encoded'].tobytes().hex()}\n")

    print(f"[Huffman/hex] Saved → {hex_path}  ({hex_path.stat().st_size / 1024:.1f} KB)")


def load_csr_from_npz(npz_path: str):
    import numpy as np
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")
    return np.load(str(npz_path), allow_pickle=True)