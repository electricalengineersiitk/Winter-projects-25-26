from compression.linear import modified_linear
from compression.conv2d import modified_conv2d

def quantize_model(model, k_clusters):
    for name, module in model.named_modules():
        if isinstance(module, (modified_linear, modified_conv2d)):
            module.quantize(k_clusters)
            module.mode = 'quantize'
    print(f"Model quantized with k={k_clusters} clusters")
