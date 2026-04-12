import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Input: 3 channels, 32x32 images
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # -> 16x32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                # -> 16x16x16
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),# -> 32x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)                 # -> 32x8x8
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(32 * 8 * 8, 128), # 32 channels * 8 width * 8 height = 2048
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_cnn(num_classes=10):
    model = SimpleCNN(num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n--- CNN Architecture ---")
    print(model)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return model