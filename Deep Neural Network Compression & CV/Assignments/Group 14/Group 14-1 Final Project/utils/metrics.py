import torch

def check_sparsity(masks):
    total_params = 0
    active_params = 0

    for mask in masks.values():
        total_params += mask.numel()
        active_params += torch.sum(mask).item()

    sparsity = 100 - (100 * active_params / total_params)
    print(f"Network Sparsity: {sparsity:.2f}% ({int(active_params)} / {total_params} active weights)")