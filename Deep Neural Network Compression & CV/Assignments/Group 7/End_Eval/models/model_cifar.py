import torch
import torch.nn as nn
from compression.linear import modified_linear
from compression.conv2d import modified_conv2d


class SmallCIFARNet(nn.Module):
    def __init__(self, num_classes=257):
        super().__init__()

        # ── CNN for LBP (1 × 100 × 100) ─────────────────────────────────
        self.lbp_cnn = nn.Sequential(
            modified_conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 100 → 50

            modified_conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 50 → 25

            modified_conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 25 → 12
        )
        # Flat: 64 × 12 × 12 = 9216

        # ── CNN for Canny (1 × 100 × 100) ───────────────────────────────
        self.canny_cnn = nn.Sequential(
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
        # Flat: 64 × 12 × 12 = 9216

        self.flatten = nn.Flatten()

        # ── Handcrafted feature encoder (12 → 128) ───────────────────────
        self.feature_encoder = nn.Sequential(
            modified_linear(12, 64),
            nn.ReLU(inplace=True),
            modified_linear(64, 128),
            nn.ReLU(inplace=True),
        )

        # ── MLP: 9216 + 9216 + 128 = 18560 → num_classes ────────────────
        self.mlp = nn.Sequential(
            modified_linear(18560, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            modified_linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            modified_linear(512, 256),
            nn.ReLU(),

            modified_linear(256, num_classes),
        )

    def forward(self, batch):
        lbp         = batch["lbp"]            # (B, 1, 100, 100)
        canny       = batch["canny"]          # (B, 1, 100, 100)
        color_feats = batch["color_features"] # (B, 6)
        shape_feats = batch["shape_features"] # (B, 6)

        lbp_out   = self.flatten(self.lbp_cnn(lbp))    # (B, 9216)
        canny_out = self.flatten(self.canny_cnn(canny)) # (B, 9216)

        hand_feats = torch.cat([color_feats, shape_feats], dim=1) # (B, 12)
        feat_out   = self.feature_encoder(hand_feats)             # (B, 128)

        fused = torch.cat([lbp_out, canny_out, feat_out], dim=1)  # (B, 18560)
        return self.mlp(fused)                                     # (B, 257)

    def prune(self, threshold):
        for m in self.modules():
            if isinstance(m, (modified_conv2d, modified_linear)):
                m.prune(threshold)

    def quantize(self, k):
        for m in self.modules():
            if isinstance(m, (modified_conv2d, modified_linear)):
                m.quantize(k)


def cifar_model(num_classes=257):
    return SmallCIFARNet(num_classes=num_classes)