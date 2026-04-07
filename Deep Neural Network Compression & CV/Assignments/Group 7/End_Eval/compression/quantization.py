import math
from .conv2d import modified_conv2d
from .linear import modified_linear


def quantize_model(model, k=16):
    bits = math.ceil(math.log2(k)) if k > 1 else 1
    for module in model.modules():
        if isinstance(module, (modified_conv2d, modified_linear)):
            module.quantize(k)
    print(f"[Quantization] Done — k={k} centroids "
          f"(~{bits}-bit, exact={math.log2(k):.2f}-bit)")