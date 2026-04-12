# compression/__init__.py

# The '.' means "import from the current directory"
from .prune import prune_model
from .quantization import quantize_model
from .linear import modified_linear
from .conv2d import modified_conv2d

# __all__ tells Python exactly what is available to be imported from this package
__all__ = [
    "prune_model",
    "quantize_model",
    "modified_linear",
    "modified_conv2d"
]