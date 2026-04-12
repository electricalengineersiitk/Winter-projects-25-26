
# Basically this is the MLP architecture for MNIST dataset we have taken. 
# It consists of 4 fully connected layers with ReLU activation function in between. 
# The final layer outputs the class probabilities for the 10 classes in MNIST. 
# The model also includes methods for pruning and quantization, which will be implemented later.


import torch
import torch.nn as nn
from compression.linear import modified_linear

class mnist(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            modified_linear(28*28, 256),
            nn.ReLU(),

            modified_linear(256, 512),
            nn.ReLU(),

            modified_linear(512, 256),
            nn.ReLU(),

            modified_linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    # To all layers
    def prune(self, ratio):
        for module in self.modules():
            if isinstance(module, modified_linear):
                module.prune(ratio)


    # For quantization (later)
    def quantize(self, k):
        for module in self.modules():
            if isinstance(module, modified_linear):
                module.quantize(k)