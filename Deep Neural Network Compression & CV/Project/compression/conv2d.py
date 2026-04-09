import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

class modified_conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        mode="normal",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device=device,
            dtype=dtype,
        )
        self.mode = mode
        self.register_buffer("mask", torch.ones_like(self.weight))
        self.register_buffer(
            "assignments",
            torch.zeros_like(self.weight, dtype=torch.long),
        )
        self.hashtable = None 

    def prune(self, threshold):
        self.mode = "prune"
        new_mask = (self.weight.abs() > threshold).float()
        self.mask.mul_(new_mask)  
        print("pruned_conv2d!!")
        return

    def quantize(self, k):
        self.mode = "quantize"
        print("quantized_conv2d")
        matrix = self.weight.detach().cpu().numpy()
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )
        kmeans.fit(matrix.reshape(-1, 1))
        # assignments = torch.from_numpy(
        #     kmeans.labels_
        # ).reshape(self.weight.shape).long() #original code
        assignments = (
            torch.from_numpy(kmeans.labels_)
            .reshape(self.weight.shape)
            .to(torch.uint8)
        )
        self.assignments.copy_(assignments)
        self.hashtable = nn.Parameter(
            torch.from_numpy(
                kmeans.cluster_centers_.reshape(-1)
            ).to(self.weight.device, self.weight.dtype)
        )
        self.weight.requires_grad = False
        with torch.no_grad():
            self.weight.data = torch.empty(0, device=self.weight.device)
        return

    def forward(self, input):
        if self.mode == "normal":
            W = self.weight
        elif self.mode == "prune":
            W = self.weight * self.mask
        elif self.mode == "quantize":
            W = self.hashtable[self.assignments] * self.mask
        return F.conv2d(
            input,
            W,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )