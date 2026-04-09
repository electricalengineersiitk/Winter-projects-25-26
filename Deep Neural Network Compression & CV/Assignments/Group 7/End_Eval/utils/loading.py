import torch
import numpy as np


def save_model_npz(model, path):
    arrays = {}
    for name, param in model.named_parameters():
        arrays[f"param_{name}"] = param.data.cpu().numpy()
    for name, buf in model.named_buffers():
        arrays[f"buffer_{name}"] = buf.cpu().numpy()
    np.savez_compressed(path, **arrays)
    print(f"[Saved] {path}")


def load_model_from_npz(model, path, device):
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    for name, param in model.named_parameters():
        key = f"param_{name}"
        if key in keys:
            param.data = torch.tensor(
                data[key].astype(np.float32), dtype=param.dtype
            ).to(device)
        else:
            print(f"[Load] WARNING: missing param '{name}' in {path}")

    for name, buf in model.named_buffers():
        key = f"buffer_{name}"
        if key in keys:
            buf.copy_(
                torch.tensor(data[key], dtype=buf.dtype).to(device)
            )
        else:
            print(f"[Load] WARNING: missing buffer '{name}' in {path}")

    model.to(device)
    print(f"[Loaded] {path}")
    return model