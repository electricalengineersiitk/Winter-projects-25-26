import os
import torch
import kagglehub
from data import create_loaders
from models import get_dnn_model
from compression import apply_pruning, check_sparsity, apply_quantization
from utils import train_and_eval, test_eval, save_model_npz, load_model_from_npz
from config import get_compute_device

def run_pipline():
    src_path = kagglehub.dataset_download("moltean/fruits")
    base_dir = os.path.join(src_path, "fruits-360_100x100", "fruits-360")
    tr_path, te_path = os.path.join(base_dir, "Training"), os.path.join(base_dir, "Test")
    
    loader_tr, loader_te = create_loaders(tr_path, te_path)
    
    batch_sample = next(iter(loader_tr))
    print("\n[Data] Shapes:")
    for k, v in batch_sample.items():
        print(f"  {k}: {tuple(v.shape)}")

    hw_ctx = get_compute_device()
    if hw_ctx.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
    net = get_dnn_model().to(hw_ctx)
    total_p = sum(p.numel() for p in net.parameters())
    print(f"\n[Model] Params count: {total_p:,}")

    print("\n--- Baseline ---")
    train_and_eval(net, loader_tr, loader_te, hw_ctx, epochs=10)
    os.makedirs("output_models", exist_ok=True)
    torch.save(net.state_dict(), "output_models/init.pth")

    print("\n--- Pruning ---")
    apply_pruning(net, t=0.05)
    check_sparsity(net)
    train_and_eval(net, loader_tr, loader_te, hw_ctx, epochs=5)
    torch.save(net.state_dict(), "output_models/pruned_net.pth")

    print("\n--- Quantization ---")
    apply_quantization(net, k=16)
    train_and_eval(net, loader_tr, loader_te, hw_ctx, epochs=5)
    torch.save(net.state_dict(), "output_models/quantized_net.pth")

    save_model_npz(net, "output_models/final_compressed.npz")

    
    net_restore = get_dnn_model().to(hw_ctx)

    net_restore = load_model_from_npz(net_restore, "output_models/final_compressed.npz", hw_ctx)
    test_eval(net_restore, loader_te, hw_ctx)

    def calc_mb(f):
        return os.path.getsize(f) / (1024**2)

    sz_base = calc_mb("output_models/init.pth")
    sz_prun = calc_mb("output_models/pruned_net.pth")
    sz_quan = calc_mb("output_models/quantized_net.pth")
    sz_comp = calc_mb("output_models/final_compressed.npz")

    print("\nFinal Metrics:")
    print(f"  Initial:   {sz_base:.2f} MB")
    print(f"  Pruned:    {sz_prun:.2f} MB ({100*(1-sz_prun/sz_base):.1f}% reduction)")
    print(f"  Quantized: {sz_quan:.2f} MB ({100*(1-sz_quan/sz_base):.1f}% reduction)")
    print(f"  Compressed:{sz_comp:.2f} MB ({100*(1-sz_comp/sz_base):.1f}% reduction)")

if __name__ == '__main__':
    run_pipline()     