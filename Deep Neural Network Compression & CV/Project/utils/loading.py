import numpy as np
from compression.linear import modified_linear
from compression.conv2d import modified_conv2d
from scipy.sparse import csr_matrix
import numpy as np
import torch
from compression.linear import modified_linear
from compression.conv2d import modified_conv2d

def save_model_npz(model, savepath):
    arrays = {}
    count = 0

    for name, module in model.named_modules():
        if not hasattr(module, "assignments"):
            continue
        if not hasattr(module, "hashtable"):
            continue
        if module.hashtable is None:
            continue

        count += 1
        prefix = name.replace(".", "_")

        arrays[f"{prefix}__shape"] = np.array(module.mask.shape)
        arrays[f"{prefix}__mask"] = module.mask.detach().cpu().numpy().astype(np.bool_)
        arrays[f"{prefix}__assignments"] = (
            module.assignments.detach().cpu().numpy().astype(np.uint8)
        )
        arrays[f"{prefix}__codebook"] = (
            module.hashtable.detach().cpu().numpy().astype(np.float32)
        )

        if module.bias is not None:
            arrays[f"{prefix}__bias"] = (
                module.bias.detach().cpu().numpy().astype(np.float32)
            )

    print(f"[save_model_npz] saved {count} layers")

    if count == 0:
        raise RuntimeError("NO COMPRESSED LAYERS FOUND")

    np.savez_compressed(savepath, **arrays)


def load_model_from_npz(model, npz_path, device="cpu"):
    raw = np.load(npz_path, allow_pickle=True)

    for name, module in model.named_modules():
        if not isinstance(module, (modified_linear, modified_conv2d)):
            continue

        prefix = name.replace(".", "_")

        key_mask = f"{prefix}__mask"
        if key_mask not in raw:
            continue   # layer not compressed

        # ---- LOAD DATA ----
        mask = torch.from_numpy(raw[f"{prefix}__mask"]).to(device)
        assignments = torch.from_numpy(raw[f"{prefix}__assignments"]).to(device)
        codebook = torch.from_numpy(raw[f"{prefix}__codebook"]).to(device)
        bias = (
            torch.from_numpy(raw[f"{prefix}__bias"]).to(device)
            if f"{prefix}__bias" in raw else None
        )

        # ---- COPY INTO MODULE ----
        module.mask.copy_(mask)
        module.assignments.copy_(assignments)

        module.hashtable = torch.nn.Parameter(
            codebook, requires_grad=False
        )

        if bias is not None and module.bias is not None:
            module.bias.data.copy_(bias)

        # ---- KILL DENSE WEIGHTS ----
        module.weight = None
        module.mode = "quantize"

        return model
    
def load_csr_from_npz(npz_path):
    raw = np.load(npz_path)
    csr_layers = {}

    # detect unique layer prefixes
    prefixes = set(k.split("__")[0] for k in raw.files)

    for prefix in prefixes:
        shape = tuple(raw[f"{prefix}__shape"])

        mask = raw[f"{prefix}__mask"]
        assignments = raw[f"{prefix}__assignments"]
        codebook = raw[f"{prefix}__codebook"]
        bias = raw[f"{prefix}__bias"] if f"{prefix}__bias" in raw else None

        # ---- Dense layer ----
        if len(shape) == 2:
            mask_2d = mask
            assign_2d = assignments
            out_shape = shape

        # ---- Conv2d layer ----
        elif len(shape) == 4:
            oc, ic, kh, kw = shape
            mask_2d = mask.reshape(oc, -1)
            assign_2d = assignments.reshape(oc, -1)
            out_shape = (oc, ic * kh * kw)

        else:
            continue

        rows, cols = np.nonzero(mask_2d)
        if rows.size == 0:
            continue

        csr = csr_matrix(
            (assign_2d[rows, cols], (rows, cols)),
            shape=out_shape
        )

        csr_layers[prefix] = {
            "csr": csr,
            "codebook": codebook,
            "bias": bias,
            "orig_shape": shape
        }

    return csr_layers
