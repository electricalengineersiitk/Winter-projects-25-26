import torch
import numpy as np
import os
from compression.conv2d import modified_conv2d
from compression.linear import modified_linear

def save_model_npz(model, path):
    """
    Saves only the non-zero weights and masks to a compressed .npz file.
    This is what gives you the massive disk space reduction.
    """
    data_dict = {}
    
    for name, m in model.named_modules():
        if isinstance(m, (modified_conv2d, modified_linear)):
            # Convert to CPU numpy for storage
            weights = m.weight.data.cpu().numpy()
            mask = m.mask.cpu().numpy().astype(bool)
            
            # Save only the values that aren't 0
            compressed_weights = weights[mask]
            data_dict[f"{name}.weights"] = compressed_weights
            data_dict[f"{name}.mask"] = mask
        
        # Save standard buffers
        elif hasattr(m, 'weight') and m.weight is not None:
             data_dict[f"{name}.weight"] = m.weight.data.cpu().numpy()
             if hasattr(m, 'bias') and m.bias is not None:
                 data_dict[f"{name}.bias"] = m.bias.data.cpu().numpy()

    np.savez_compressed(path, **data_dict)
    print(f"✅ Compressed model saved to {path}")

def load_model_from_npz(model, path, device):
    """
    Reconstructs the model from the compressed .npz file.
    """
    checkpoint = np.load(path)
    
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, (modified_conv2d, modified_linear)):
                weights_key = f"{name}.weights"
                mask_key = f"{name}.mask"
                
                if weights_key in checkpoint:
                    mask = checkpoint[mask_key]
                    compressed_weights = checkpoint[weights_key]
                    
                    # Reconstruct the full matrix
                    full_weights = np.zeros(mask.shape, dtype=np.float32)
                    full_weights[mask] = compressed_weights
                    
                    m.weight.data.copy_(torch.from_numpy(full_weights).to(device))
                    m.mask.copy_(torch.from_numpy(mask.astype(np.float32)).to(device))
            
            elif hasattr(m, 'weight') and m.weight is not None:
                weight_key = f"{name}.weight"
                if weight_key in checkpoint:
                    m.weight.data.copy_(torch.from_numpy(checkpoint[weight_key]).to(device))
    
    print(f"Model loaded from {path}")
    return model