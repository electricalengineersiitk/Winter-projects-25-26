# models/model_cifar.py — miniVGG architecture for CIFAR-10

import torch.nn as nn


class miniVGG(nn.Module):
    """
    VGG-style CNN for CIFAR-10 (32x32 input).

    Architecture:
        Block 1 : Conv(3→32)  + BN + ReLU + MaxPool  → 16x16
        Block 2 : Conv(32→64) + BN + ReLU + MaxPool  → 8x8
        Block 3 : Conv(64→128) + BN + ReLU
                  Conv(128→128) + BN + ReLU + MaxPool → 4x4
        Block 4 : Conv(128→256) + BN + ReLU
                  AdaptiveAvgPool                     → 2x2
        Classifier : Linear(1024→512) + BN + ReLU + Dropout(0.4)
                     Linear(512→10)
    """

    def __init__(self):
        super(miniVGG, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 32x32 → 16x16
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 16x16 → 8x8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 8x8 → 4x4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 4x4 → 2x2
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),   # output: 2x2x256 = 1024 features
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
