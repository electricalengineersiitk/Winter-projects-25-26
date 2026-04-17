from .model_cifar import SmallCIFARNet, cifar_model

__all__ = [
    "SmallCIFARNet",  # Model class — used for isinstance checks in compression
    "cifar_model",    # Factory function → SmallCIFARNet(num_classes=237)
]