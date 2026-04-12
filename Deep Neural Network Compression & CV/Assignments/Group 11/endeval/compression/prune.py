from .conv2d import modified_conv2d
from .linear import modified_linear

def apply_pruning(net, t=0.05):
    for m in net.modules():
        if isinstance(m, (modified_conv2d, modified_linear)):
            m.prune(t)
    print(f"[Pruning] Done — t={t}")

def check_sparsity(net):
    tot = 0
    nil = 0
    for m in net.modules():
        if isinstance(m, (modified_conv2d, modified_linear)):
            tot += m.mask.numel()
            nil += (m.mask == 0).sum().item()
    s = 100.0 * nil / tot if tot > 0 else 0.0
    print(f"[Sparsity] {nil}/{tot} ({s:.2f}%)")
    return s