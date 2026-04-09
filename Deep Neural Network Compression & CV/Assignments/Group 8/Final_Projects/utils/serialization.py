import numpy as np
import torch
import os   

def save_compressed_model(model, path):
    import numpy as np

    data = {}
    layer_id = 0

    for module in model.modules():
        if hasattr(module, "weight"):
            w = module.weight.data.cpu().numpy()

            # 🔥 Convert to int8
            w_int8 = w.astype(np.int8)

            data[f"layer_{layer_id}_weight"] = w_int8

            if module.bias is not None:
                data[f"layer_{layer_id}_bias"] = module.bias.data.cpu().numpy()

            layer_id += 1

    np.savez_compressed(path, **data)