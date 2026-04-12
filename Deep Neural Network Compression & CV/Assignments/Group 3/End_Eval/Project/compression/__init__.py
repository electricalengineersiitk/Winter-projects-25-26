#make the main compression tools easy to access from the root of the package

from .pruning import prune_model, apply_pruning_masks
from .quantization import quantize_model, quantize_and_finetune
from .huffman import huffman_encode_model, compute_huffman_savings
