import torch
from torch import nn
from compression.linear import modified_linear
from compression.conv2d import modified_conv2d

class SmallCIFARNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
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
            modified_linear(128 * 4 * 4, 256),
            nn.ReLU(),
            modified_linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def prune(self, sparsity=0.90):
        """
        Stage 1: Applies L1 unstructured pruning to enforce sparsity.
        Iterates through the custom compressible layers to remove the specified 
        percentage of weights (default 90%).
        """
        for name, module in self.named_modules():
            if isinstance(module, (modified_conv2d, modified_linear)):
                if hasattr(module, 'prune'):
                    module.prune(sparsity)
                else:
                    print(f"Warning: {name} does not have a prune() method.")

    def quantize(self, k):
        """
        Stage 2: Applies K-Means quantization to cluster the remaining non-zero weights.
        Iterates through the custom compressible layers.
        """
        for name, module in self.named_modules():
            if isinstance(module, (modified_conv2d, modified_linear)):
                if hasattr(module, 'quantize'):
                    module.quantize(k)
                else:
                    print(f"Warning: {name} does not have a quantize() method.")