from compression.linear import modified_linear
from compression.conv2d import modified_conv2d

def prune_model(model, sparsity_threshold):
    for name, module in model.named_modules():
        if isinstance(module, (modified_linear, modified_conv2d)):
            module.prune(sparsity_threshold)
            module.mode = 'prune'
    print(f"Model pruned with threshold: {sparsity_threshold}")

def count_mask_sparsity(model):
    total = 0
    zeros = 0
    for module in model.modules():
        if hasattr(module, 'mask') and module.mask is not None:
            total += module.mask.numel()
            zeros += (module.mask == 0).sum().item()
    if total > 0:
        print(f"Verified Masked Sparsity: {100 * zeros / total:.2f}%")
