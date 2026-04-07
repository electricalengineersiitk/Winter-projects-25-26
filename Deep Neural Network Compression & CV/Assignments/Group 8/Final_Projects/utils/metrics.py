
import torch

def calculate_sparsity(model):
    total = 0
    zero = 0

    for module in model.modules():
        if hasattr(module, 'mask'):
            total += module.mask.numel()
            zero += (module.mask == 0).sum().item()

    sparsity = 100 * zero / total
    return sparsity


def count_unique_weights(model):
    unique_vals = set()

    for module in model.modules():
        if hasattr(module, "weight"):
            unique_vals.update(module.weight.data.cpu().numpy().flatten())

    return len(unique_vals)


def detailed_sparsity_report(model):
    total = 0
    zero = 0

    for module in model.modules():
        if hasattr(module, "mask"):
            total += module.mask.numel()
            zero += (module.mask == 0).sum().item()

    active = total - zero

    print("--- PRUNING REPORT ---")
    print(f"Total Weights:  {total:,}")
    print(f"Active Weights: {active:,}")
    print(f"Zeroed Weights: {zero:,}")
    print(f"Sparsity:       {100*zero/total:.2f}%")



def quantization_report(model):
    print("\n--- QUANTIZATION STATS ---")

    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            weights = module.weight.data.cpu().numpy().flatten()
            unique_vals = len(set(weights))

            zero_mask = 1 if hasattr(module, "mask") else 0

            print(f"Layer '{name}': {unique_vals} unique active weights + {zero_mask} zero-mask")

    print("--------------------------\n")


def calculate_storage(model, bits=32):
    total_params = 0

    for module in model.modules():
        if hasattr(module, "weight"):
            total_params += module.weight.numel()

    total_bits = total_params * bits
    total_mb = total_bits / (8 * 1024 * 1024)

    return total_mb