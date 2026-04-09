import torch
from torch import nn
from compression.linear import modified_linear

class mnist_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            modified_linear(28 * 28, 256),
            nn.ReLU(),
            modified_linear(256, 512),
            nn.ReLU(),
            modified_linear(512, 256),
            nn.ReLU(),
            modified_linear(256, 10)
        )
        
    def forward(self,x):
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def prune(self,threshold):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, modified_linear)):
                module.prune(threshold)
    
    def quantize(self,k):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, modified_linear)):
                module.quantize(k)
