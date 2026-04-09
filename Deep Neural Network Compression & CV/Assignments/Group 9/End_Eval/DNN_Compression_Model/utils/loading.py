import numpy as np
import torch
from scipy.sparse import csr_matrix

def save_model_npz(model, file_path):
    save_dict = {}
    
    for name, param in model.state_dict().items():
        is_custom_weight = 'weight' in name and any(name.startswith(m_name) for m_name, m in model.named_modules() if hasattr(m, 'mode'))
        if not is_custom_weight:
            save_dict[f"sd_{name}"] = param.cpu().numpy()

    for name, module in model.named_modules():
        if hasattr(module, 'mode') and module.mode == 'quantize':
            save_dict[f"custom_{name}.cluster_centers"] = module.cluster_centers.detach().cpu().numpy()
            
            indices_np = module.weight_indices.cpu().numpy()
            mask_np = module.mask.cpu().numpy() if module.mask is not None else np.ones_like(indices_np)
            
            safe_indices = ((indices_np + 1) * mask_np).astype(np.uint8)
            
            if len(safe_indices.shape) > 2:
                safe_indices = safe_indices.reshape(safe_indices.shape[0], -1)
                
            sparse_indices = csr_matrix(safe_indices)
            save_dict[f"custom_{name}.indices_data"] = sparse_indices.data
            save_dict[f"custom_{name}.indices_indices"] = sparse_indices.indices
            save_dict[f"custom_{name}.indices_indptr"] = sparse_indices.indptr
            save_dict[f"custom_{name}.shape"] = module.weight.shape
            
    np.savez_compressed(file_path, **save_dict)
    print(f"Model compressed and saved to {file_path}")

def load_model_from_npz(model, file_path, device):
    loaded_data = np.load(file_path)
    
    state_dict = model.state_dict()
    for key in loaded_data.files:
        if key.startswith("sd_"):
            original_name = key[3:] 
            if original_name in state_dict:
                state_dict[original_name].copy_(torch.from_numpy(loaded_data[key]))
    
    model.load_state_dict(state_dict, strict=False)
    
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            if f"custom_{name}.cluster_centers" in loaded_data:
                centers = torch.tensor(loaded_data[f"custom_{name}.cluster_centers"], device=device)
                
                data = loaded_data[f"custom_{name}.indices_data"]
                indices = loaded_data[f"custom_{name}.indices_indices"]
                indptr = loaded_data[f"custom_{name}.indices_indptr"]
                shape = tuple(loaded_data[f"custom_{name}.shape"])
                
                if len(shape) > 2:
                    flat_shape = (shape[0], np.prod(shape[1:]))
                    sparse_matrix = csr_matrix((data, indices, indptr), shape=flat_shape)
                    dense_indices = sparse_matrix.toarray()
                else:
                    sparse_matrix = csr_matrix((data, indices, indptr), shape=shape)
                    dense_indices = sparse_matrix.toarray()
                
                dense_indices = dense_indices.astype(np.int32)
                mask_np = (dense_indices > 0).astype(np.float32)
                original_indices = np.maximum(dense_indices - 1, 0)
                
                if len(shape) > 2:
                    original_indices = original_indices.reshape(shape)
                    mask_np = mask_np.reshape(shape)
                    
                idx_tensor = torch.tensor(original_indices, device=device, dtype=torch.long)
                mask_tensor = torch.tensor(mask_np, device=device, dtype=torch.float32)
                
                reconstructed_weight = centers[idx_tensor].view(shape)
                reconstructed_weight = reconstructed_weight * mask_tensor
                
                module.weight.data = reconstructed_weight
                module.mode = 'normal'
                        
    return model
