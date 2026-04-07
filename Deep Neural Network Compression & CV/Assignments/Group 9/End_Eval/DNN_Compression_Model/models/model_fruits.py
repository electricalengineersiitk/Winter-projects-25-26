import torch
import torch.nn as nn
from compression.conv2d import modified_conv2d
from compression.linear import modified_linear

class FruitsFusionNet(nn.Module):
    def __init__(self, num_classes=131):
        super().__init__()
        
        self.image_features = nn.Sequential(
            modified_conv2d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            modified_conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            modified_conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            modified_linear(8206, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            modified_linear(256, num_classes)
        )

    def forward(self, img, lbp, canny, shape, color):
        stacked_images = torch.cat((img, lbp, canny), dim=1)
        
        img_feats = self.image_features(stacked_images)
        img_feats = self.flatten(img_feats)
        
        fused_features = torch.cat((img_feats, shape, color), dim=1)
        
        output = self.classifier(fused_features)
        return output
