# compression subpackage - pruning, quantization, huffman encoding
from .linear import CompressedLinear,replace_linear_with_compressed
from .conv2d import CompressedConv2d,replace_conv2d_with_compressed
from .pruning import apply_pruning,compute_sparsity
from .quantization import apply_quantization
from .huffman import huffman_encode,huffman_decode
