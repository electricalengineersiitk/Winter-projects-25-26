import torch
from torch import nn

class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.flatten = nn.Flatten()
        
        
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(128, 10) # 10 output classes for digits 0-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x