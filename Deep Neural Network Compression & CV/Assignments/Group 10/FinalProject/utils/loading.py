
import torch
import numpy as np
from scipy.sparse import csr_matrix

def save_model_npz(model, path):
    """Saves model weights in a compressed sparse format to actually save disk space."""
    state_dict = model.state_dict()
    sparse_dict = {}
    
    for name, tensor in state_dict.items():
        weight_np = tensor.cpu().numpy()
        
       
        original_shape = weight_np.shape
        weight_2d = weight_np.reshape(weight_np.shape[0], -1)
        
       
        sparse_mat = csr_matrix(weight_2d)
        
       
        sparse_dict[f"{name}_data"] = sparse_mat.data
        sparse_dict[f"{name}_indices"] = sparse_mat.indices
        sparse_dict[f"{name}_indptr"] = sparse_mat.indptr
        sparse_dict[f"{name}_shape"] = np.array(original_shape)
        
    np.savez_compressed(path, **sparse_dict)
    

def load_csr_from_npz(npz_file, name):
    data = npz_file[f"{name}_data"]
    indices = npz_file[f"{name}_indices"]
    indptr = npz_file[f"{name}_indptr"]
    shape = npz_file[f"{name}_shape"]
    
    sparse_mat = csr_matrix((data, indices, indptr), shape=(shape[0], np.prod(shape[1:])))
    
    
    dense_array = sparse_mat.toarray().reshape(shape)
    return torch.from_numpy(dense_array)

def load_model_from_npz(model, path, device):
    """Loads compressed weights back into a PyTorch model."""
    npz_file = np.load(path)
    state_dict = model.state_dict()
    
    for name in state_dict.keys():
        if f"{name}_data" in npz_file:
            tensor = load_csr_from_npz(npz_file, name)
            state_dict[name] = tensor
            
    model.load_state_dict(state_dict)
    model.to(device)
    
    return model