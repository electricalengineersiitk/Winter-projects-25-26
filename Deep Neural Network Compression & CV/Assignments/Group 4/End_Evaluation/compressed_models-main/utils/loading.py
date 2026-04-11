import torch
import numpy as np
from compression.linear import ModifiedLinear

def save_model_npz(model, filepath):
    """
    Saves the compressed model parameters to an .npz file.
    Stores only necessary metadata (masks, cluster centers, indices, biases).
    """
    state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, ModifiedLinear):
            # Save biases (uncompressed usually)
            if module.bias is not None:
                state_dict[f"{name}.bias"] = module.bias.detach().cpu().numpy()
            
            # Save mask (boolean to save space)
            if hasattr(module, 'mask'):
                state_dict[f"{name}.mask"] = module.mask.detach().cpu().numpy().astype(bool)
                
            # Save quantization data
            if module.mode == 'quantized':
                if hasattr(module, 'cluster_centers'):
                    state_dict[f"{name}.cluster_centers"] = module.cluster_centers
                if hasattr(module, 'cluster_indices'):
                    # Save as uint8 or int16 depending on k to save space
                    state_dict[f"{name}.cluster_indices"] = module.cluster_indices.detach().cpu().numpy().astype(np.uint8)
            else:
                # If not quantized, just save the sparse weights
                sparse_weights = module.weight.detach().cpu().numpy()
                state_dict[f"{name}.weight"] = sparse_weights
                
        elif isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)): # Standard fallback
            state_dict[f"{name}.weight"] = module.weight.detach().cpu().numpy()
            if module.bias is not None:
                state_dict[f"{name}.bias"] = module.bias.detach().cpu().numpy()

    np.savez_compressed(filepath, **state_dict)
    print(f"Model successfully serialized to {filepath}")

def load_model_from_npz(model, filepath, device):
    """
    Loads compressed parameters from an .npz file back into the PyTorch model.
    """
    npz_file = np.load(filepath)
    
    for name, module in model.named_modules():
        if isinstance(module, ModifiedLinear):
            if f"{name}.bias" in npz_file:
                module.bias.data = torch.tensor(npz_file[f"{name}.bias"], device=device)
            
            if f"{name}.mask" in npz_file:
                module.mask = torch.tensor(npz_file[f"{name}.mask"], dtype=torch.float32, device=device)
                
            if f"{name}.cluster_centers" in npz_file and f"{name}.cluster_indices" in npz_file:
                module.cluster_centers = npz_file[f"{name}.cluster_centers"]
                module.cluster_indices = torch.tensor(npz_file[f"{name}.cluster_indices"], dtype=torch.int64, device=device)
                module.mode = 'quantized'
                
                # Reconstruct weight data
                centers_tensor = torch.tensor(module.cluster_centers, dtype=module.weight.dtype, device=device)
                reconstructed = centers_tensor[module.cluster_indices]
                module.weight.data = reconstructed * module.mask
                
            elif f"{name}.weight" in npz_file:
                module.weight.data = torch.tensor(npz_file[f"{name}.weight"], device=device)
                if hasattr(module, 'mask'):
                    module.mode = 'pruned'

    print(f"Model successfully loaded from {filepath}")
    return model

def load_csr_from_npz(filepath):
    """
    Placeholder for CSR specific loading if required by your mentors.
    Standard npz loading (above) usually suffices for this pipeline.
    """
    pass