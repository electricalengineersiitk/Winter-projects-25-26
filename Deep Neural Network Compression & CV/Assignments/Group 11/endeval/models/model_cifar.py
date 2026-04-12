import torch
import torch.nn as nn
from compression.linear import modified_linear
from compression.conv2d import modified_conv2d

class FruitModel(nn.Module):
    def __init__(self, num_classes=237):
        super().__init__()

        self.f_ext1 = nn.Sequential(
            modified_conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            modified_conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            modified_conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.f_ext2 = nn.Sequential(
            modified_conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            modified_conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            modified_conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.flat = nn.Flatten()
        f_dim = 9216 + 9216 + 6 + 6

        self.layer_stack = nn.Sequential(
            modified_linear(f_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            modified_linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            modified_linear(512, 256),
            nn.ReLU(),
            modified_linear(256, num_classes),
        )

    def forward(self, x):
        v1, v2 = x["lbp"], x["canny"]
        c_f, s_f = x["color_features"], x["shape_features"]

        o1 = self.flat(self.f_ext1(v1))
        o2 = self.flat(self.f_ext2(v2))

        fused = torch.cat([o1, o2, c_f, s_f], dim=1)
        return self.layer_stack(fused)

    def prune(self, thresh):
        for m in self.modules():
            if isinstance(m, (modified_conv2d, modified_linear)):
                m.prune(thresh)

    def quantize(self, k):
        for m in self.modules():
            if isinstance(m, (modified_conv2d, modified_linear)):
                m.quantize(k)

def get_dnn_model():
    return FruitModel(num_classes=237)