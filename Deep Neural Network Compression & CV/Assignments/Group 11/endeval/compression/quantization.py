import math
from .conv2d import modified_conv2d
from .linear import modified_linear

def apply_quantization(net, k=16):
    b = math.ceil(math.log2(k)) if k > 1 else 1
    for m in net.modules():
        if isinstance(m, (modified_conv2d, modified_linear)):
            m.quantize(k)

    print(f"[Quantization] {k} clusters (~{b} bits)")