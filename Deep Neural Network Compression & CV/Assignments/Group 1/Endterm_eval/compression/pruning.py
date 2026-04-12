# magnitude-based weight pruning
# the idea is simple: small weights dont contribute much to the output, so we can zero them out
# steps:
#   1. compute |w| for every weight in the model
#   2. sort all weights globally by magnitude
#   3. zero out the smallest fraction (the sparsity %) using a binary mask
#   4. fine-tune the surviving weights to recover accuracy
# this is from the deep compression paper (Han et al., 2016)


import torch
import numpy as np
from .linear import CompressedLinear
from .conv2d import CompressedConv2d

def create_mask(weight_tensor,threshold):
    mask=(weight_tensor.abs()>=threshold).float()
    return mask


def compute_threshold_for_sparsity(weight_tensor,target_sparsity):
    magnitudes=weight_tensor.abs().flatten()
    sorted_mags,_=torch.sort(magnitudes)
    cutoff_index=int(target_sparsity*len(sorted_mags))
    cutoff_index=min(cutoff_index,len(sorted_mags)-1) 
    threshold=sorted_mags[cutoff_index].item()
    return threshold


def apply_pruning(model,target_sparsity=0.6):
    print(f"\n--- Applying Pruning (target sparsity: {target_sparsity:.0%}) ---")

    all_magnitudes=[]
    compressed_layers=[]

    for name,module in model.named_modules():
        if isinstance(module, (CompressedLinear, CompressedConv2d)):
            compressed_layers.append((name,module))
            active_weights=module.weight.data*module.mask
            all_magnitudes.append(active_weights.abs().flatten())

    if not compressed_layers:
        print("  No CompressedLinear layers found. Run replace_linear first.")
        return model

    all_mags=torch.cat(all_magnitudes)
    threshold=compute_threshold_for_sparsity(all_mags,target_sparsity)
    print(f"  Global magnitude threshold: {threshold:.6f}")

    total_params=0
    total_pruned=0

    for name,module in compressed_layers:
        active_weights=module.weight.data*module.mask
        new_mask=create_mask(active_weights,threshold)
        combined_mask=module.mask*new_mask
        module.set_mask(combined_mask)

        # FIX: Explicitly zero out the actual weight values
        # This ensures underlying weight data strictly respects the mask
        module.weight.data *= combined_mask

        layer_total=module.weight.numel()
        layer_pruned=(combined_mask==0).sum().item()
        total_params+=layer_total
        total_pruned+=layer_pruned

        print(f"  Layer '{name}': "
              f"{layer_pruned}/{layer_total} pruned "
              f"({100*layer_pruned/layer_total:.1f}%)")

    overall_sparsity=total_pruned/total_params
    print(f"\n  Overall sparsity: {overall_sparsity:.2%}")
    print(f"  Active parameters: {total_params-total_pruned:,} "
          f"out of {total_params:,}")

    return model


def compute_sparsity(model):
    sparsity_dict={}
    for name,module in model.named_modules():
        if isinstance(module, (CompressedLinear, CompressedConv2d)):
            sparsity=module.get_sparsity()
            sparsity_dict[name]=sparsity

    return sparsity_dict