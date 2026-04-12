import torch
from torch import nn
from compression.linear import modified_linear
from compression.conv2d import modified_conv2d

class SmallCIFARNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            modified_conv2d(3, 32, 3, padding=1),
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
        self.softmax = nn.Softmax(dim = 1)
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

    def prune(self,threshold):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.prune(threshold)
    
    def quantize(self,k):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.quantize(k)