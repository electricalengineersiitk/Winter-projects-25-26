from .linear import ModifiedLinear, modified_linear
from .conv2d import ModifiedConv2d, modified_conv2d
from .pruning import prune_model, apply_masks, count_pruned_weights, get_layer_sparsities
from .quantization import quantize_model, quantize_layer, get_quantization_stats
from .huffman import huffman_encode_model, print_compression_summary, get_compressed_size_bits