# saving and loading compressed models in NPZ format
# universal version: handles flat, 2D (Linear), and 4D (Conv2d) tensors automatically

import numpy as np
import os
import torch

from compression.linear import CompressedLinear
from compression.conv2d import CompressedConv2d


def save_compressed_npz(model,filepath,quantization_info=None,
                        huffman_info=None):
    # save the compressed model to .npz file
    save_dict={}
    layer_idx=0

    for name,module in model.named_modules():
        if isinstance(module, (CompressedLinear, CompressedConv2d)):
            prefix=f'layer_{layer_idx}'

            weight_np=module.weight.data.cpu().numpy()
            mask_np=module.mask.data.cpu().numpy()

            # save the original shape so we can reconstruct it later
            save_dict[f'{prefix}_shape']=np.array(weight_np.shape)

            # FIX: Flatten the arrays to handle any dimension (2D or 4D)
            weight_flat = weight_np.flatten()
            mask_flat = mask_np.flatten()

            # save mask as sparse 1D indices instead of full matrix
            nonzero_indices = np.where(mask_flat > 0)[0]
            save_dict[f'{prefix}_mask_indices'] = nonzero_indices.astype(np.int32)

            # only save the non-zero weights
            nonzero_weights = weight_flat[nonzero_indices]
            save_dict[f'{prefix}_weights'] = nonzero_weights.astype(np.float16)

            # bias
            if module.bias is not None:
                save_dict[f'{prefix}_bias']=module.bias.data.cpu().numpy()

            # quantization info
            if quantization_info and name in quantization_info:
                q_info=quantization_info[name]
                save_dict[f'{prefix}_centroids']=q_info['centroids'].astype(np.float16)
                indices=q_info['indices']
                nonzero_mask=mask_flat>0
                save_dict[f'{prefix}_q_indices']=indices[nonzero_mask].astype(np.uint8)

            layer_idx+=1

    save_dict['num_layers']=np.array([layer_idx])

    if huffman_info:
        for layer_name,h_info in huffman_info.items():
            save_dict[f'huffman_{layer_name}_bits']=np.frombuffer(
                h_info['encoded_bits'].encode('ascii'),dtype=np.uint8)
            for symbol,code in h_info['codebook'].items():
                save_dict[f'huffman_{layer_name}_code_{symbol}']=\
                    np.array([ord(c) for c in code],dtype=np.uint8)

    np.savez_compressed(filepath,**save_dict)

    file_size=os.path.getsize(filepath)
    print(f"\nCompressed model saved to: {filepath}")
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

    return file_size


def save_original_npz(model,filepath):
    # save the original uncompressed model for size comparison
    save_dict={}
    layer_idx=0

    for name,module in model.named_modules():
        if hasattr(module,'weight') and isinstance(module.weight,torch.nn.Parameter):
            prefix=f'layer_{layer_idx}'
            save_dict[f'{prefix}_weight']=module.weight.data.cpu().numpy()
            if hasattr(module,'bias') and module.bias is not None:
                save_dict[f'{prefix}_bias']=module.bias.data.cpu().numpy()
            layer_idx+=1

    np.savez(filepath,**save_dict)

    file_size=os.path.getsize(filepath)
    print(f"\nOriginal model saved to: {filepath}")
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

    return file_size


def load_compressed_npz(filepath):
    # load a compressed model back from .npz file
    data=np.load(filepath,allow_pickle=True)
    num_layers=data['num_layers'][0]

    model_data={}
    for i in range(num_layers):
        prefix=f'layer_{i}'
        layer_data={}

        # GET SHAPE AND TOTAL ELEMENTS
        shape=tuple(data[f'{prefix}_shape'])
        layer_data['shape']=shape
        flat_size = np.prod(shape)

        # RECONSTRUCT MASK FROM FLAT INDICES
        mask_flat=np.zeros(flat_size, dtype=np.float32)
        indices=data[f'{prefix}_mask_indices']
        mask_flat[indices]=1.0
        layer_data['mask']=mask_flat.reshape(shape) # Reshape back to 2D/4D

        # RECONSTRUCT WEIGHTS
        weights_flat=np.zeros(flat_size, dtype=np.float32)
        nonzero_vals=data[f'{prefix}_weights'].astype(np.float32)
        weights_flat[indices]=nonzero_vals
        layer_data['weights']=weights_flat.reshape(shape) # Reshape back to 2D/4D

        # bias
        bias_key=f'{prefix}_bias'
        if bias_key in data:
            layer_data['bias']=data[bias_key]

        # quantization info
        centroids_key=f'{prefix}_centroids'
        if centroids_key in data:
            layer_data['centroids']=data[centroids_key].astype(np.float32)
            indices_key=f'{prefix}_q_indices'
            if indices_key in data:
                layer_data['q_indices']=data[indices_key]

        model_data[f'layer_{i}']=layer_data

    return model_data