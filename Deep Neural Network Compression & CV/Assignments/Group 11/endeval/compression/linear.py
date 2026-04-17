import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class modified_linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

        # ── Fix Problem 3: mode saved as buffer not plain string ─────────
        # 0 = normal, 1 = prune, 2 = quantize
        self.register_buffer('mask', torch.ones_like(self.weight))
        self.register_buffer('mode_flag', torch.tensor(0))  # ← saved in .pth

    @property
    def mode(self):
        return {0: 'normal', 1: 'prune', 2: 'quantize'}[self.mode_flag.item()]

    @mode.setter
    def mode(self, value):
        self.mode_flag.fill_({'normal': 0, 'prune': 1, 'quantize': 2}[value])

    def prune(self, threshold):
        self.mask = (torch.abs(self.weight.data) >= threshold).float()
        self.weight.data *= self.mask
        self.mode = 'prune'

    def quantize(self, k):
        weights  = self.weight.data.cpu().numpy()
        mask     = self.mask.cpu().numpy()
        non_zero = weights[mask == 1].reshape(-1, 1)

        if len(non_zero) < k:
            return

        kmeans    = KMeans(n_clusters=k, random_state=0, n_init='auto')
        kmeans.fit(non_zero)
        labels    = kmeans.labels_
        centroids = kmeans.cluster_centers_.flatten()

        quantized         = weights.copy()
        quantized[mask==1] = centroids[labels]
        self.weight.data  = torch.tensor(
            quantized, dtype=self.weight.dtype
        ).to(self.weight.device)
        self.mode = 'quantize'

    def forward(self, input):
        if self.mode == 'normal':
            return F.linear(input, self.weight, self.bias)
        elif self.mode in ('prune', 'quantize'):
            return F.linear(input, self.weight * self.mask, self.bias)
        else:
            # ── Fix Problem 6: no silent None return ─────────────────────
            raise ValueError(f"[modified_linear] Unknown mode: '{self.mode}'")