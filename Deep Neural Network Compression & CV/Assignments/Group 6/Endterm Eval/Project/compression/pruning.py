import torch
import torch.nn.utils.prune as prune

def apply_magnitude_pruning(model, pruning_ratio: float):
   
    print(f"[*] Applying Magnitude Pruning (Target Sparsity: {pruning_ratio * 100:.1f}%)")
    
   
    for name, module in model.named_modules():
       
        if isinstance(module, torch.nn.Linear):
           
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            
    return model

def make_pruning_permanent(model):
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass 
    return model


def print_sparsity_stats(model, phase_name="Model Stats"):
    
    total_zeros = 0
    total_weights = 0
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            
            total_zeros += torch.sum(module.weight == 0).item()
            
            total_weights += module.weight.nelement()
            
    active_weights = total_weights - total_zeros
    sparsity = (total_zeros / total_weights) * 100 if total_weights > 0 else 0
    
    print(f"--- {phase_name} ---")
    print(f"Total Weights:  {total_weights:,}")
    print(f"Active Weights: {active_weights:,}")
    print(f"Zeroed Weights: {total_zeros:,}")
    print(f"Sparsity:       {sparsity:.2f}%\n")

def get_global_sparsity(model) -> float:
    
    total_zeros = 0
    total_weights = 0
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            
            total_zeros += torch.sum(module.weight == 0).item()
            total_weights += module.weight.nelement()
            
    if total_weights == 0:
        return 0.0
        
    return total_zeros / total_weights
def print_memory_footprint(model, bits=32, phase_name="Model"):
    
    total_zeros = 0
    total_weights = 0
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            total_zeros += torch.sum(module.weight == 0).item()
            total_weights += module.weight.nelement()
            
   
    active_weights = total_weights - total_zeros
    
    # Formula: (Weights * Bits) / 8 bits per byte / 1024 bytes per KB / 1024 KB per MB
    size_in_mb = (active_weights * bits) / (8 * 1024 * 1024)
    
    print(f"   [-] {phase_name} Storage Size: {size_in_mb:.4f} MB ({bits}-bit)")
    return size_in_mb