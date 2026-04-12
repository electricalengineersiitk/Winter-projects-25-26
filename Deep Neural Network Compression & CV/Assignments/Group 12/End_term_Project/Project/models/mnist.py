# models/mnist.py

import torch.nn as nn
from compression.linear import modified_linear
from compression.conv2d import modified_conv2d


class SmallCIFARNet(nn.Module):
    def __init__(self, num_classes: int = 131, input_size: int = 100):
        super().__init__()

        pooled = input_size // 2 // 2 // 2   # = 12
        flat   = 128 * pooled * pooled        # = 18432

        self.features = nn.Sequential(
            # Block 1
            modified_conv2d(3,  32, 3, padding=1), nn.ReLU(),
            modified_conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25),              # ← ADD THIS

            # Block 2
            modified_conv2d(32, 64, 3, padding=1), nn.ReLU(),
            modified_conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25),              # ← ADD THIS

            # Block 3
            modified_conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25),              # ← ADD THIS
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            modified_linear(flat, 256), nn.ReLU(),
            nn.Dropout(p=0.5),                 # ← ADD THIS (higher for FC)
            modified_linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def mnist_model(num_classes: int = 131, input_size: int = 100) -> SmallCIFARNet:
    return SmallCIFARNet(num_classes=num_classes, input_size=input_size)