import torch
from torch import nn
from compression.linear import ModifiedLinear

class CompressionMLP(nn.Module):
    """
    Pure MLP operating exclusively on fixed-dimensional feature vectors.
    No convolutional layers are trained or modified here.
    """
    def __init__(self, input_dim=9216, num_classes=10): # 9216 is the flattened size of AlexNet features (256 * 6 * 6)
        super(CompressionMLP, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            ModifiedLinear(input_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            ModifiedLinear(4096, 1024),
            nn.ReLU(inplace=True),
            ModifiedLinear(1024, num_classes),
        )
    
    def forward(self, x):
        # x is assumed to be a flattened 1D feature vector
        x = self.classifier(x)
        return x