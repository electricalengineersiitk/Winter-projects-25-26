import torch
import numpy as np


def save_model_npz(model, path="compressed_models/compressed.npz"):
    arrays = {}

    # Save all parameters
    for name, param in model.named_parameters():
        arrays[f"param_{name}"] = param.data.cpu().numpy()

    # ── Fix Problem 5: save ALL buffers, not just masks ──────────────────
    for name, buf in model.named_buffers():
        arrays[f"buffer_{name}"] = buf.cpu().numpy()

    np.savez_compressed(path, **arrays)
    print(f"[Saved] {path}  "
          f"({len([k for k in arrays if k.startswith('param_')])} params, "
          f"{len([k for k in arrays if k.startswith('buffer_')])} buffers)")


def load_model_from_npz(model, path, device):
    data = np.load(path)

    # Restore parameters
    for name, param in model.named_parameters():
        key = f"param_{name}"
        if key in data:
            param.data = torch.tensor(
                data[key], dtype=param.dtype
            ).to(device)
        else:
            print(f"[Load] WARNING: missing param '{name}' in {path}")

    # ── Fix Problem 5: restore ALL buffers (masks + mode_flags + BN) ─────
    for name, buf in model.named_buffers():
        key = f"buffer_{name}"
        if key in data:
            buf.copy_(torch.tensor(data[key], dtype=buf.dtype).to(device))
        else:
            print(f"[Load] WARNING: missing buffer '{name}' in {path}")

    model.to(device)
    print(f"[Loaded] {path}")
    return model