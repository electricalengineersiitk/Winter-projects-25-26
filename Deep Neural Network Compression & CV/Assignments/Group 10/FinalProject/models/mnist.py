import torch
import numpy as np
from torch import nn
from compression.linear import modified_linear
from compression.conv2d import modified_conv2d

class mnist_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # CHANGED: 1 input channel for black & white MNIST images
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            modified_conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
    
            modified_conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            modified_conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            
            modified_conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
        )
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Sequential(
            # CHANGED: 28x28 pooled 3 times results in a 3x3 spatial dimension
            modified_linear(128 * 3 * 3, 256),
            nn.ReLU(),
            modified_linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def prune(self, prune_amount):
        for module in self.modules():
            if isinstance(module, (modified_conv2d, modified_linear)):
                if hasattr(module, 'prune'):
                    module.prune(prune_amount)
                    
    def quantize(self, k):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, modified_linear, modified_conv2d)):
                if hasattr(module, 'quantize'):
                    module.quantize(k)