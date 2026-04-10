# compression/conv2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CompressedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(CompressedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # We use nn.Parameter for weight so optimizer tracks it
        weight_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(*weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.register_buffer('mask', torch.ones_like(self.weight))
        
        # Initialization
        nn.init.kaiming_uniform_(self.weight, a=torch.math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / torch.math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        masked_weight = self.weight * self.mask
        return F.conv2d(x, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def set_mask(self, mask):
        self.mask.data = mask.clone().to(self.weight.device)

    def get_sparsity(self):
        total = self.mask.numel()
        pruned = (self.mask == 0).sum().item()
        return pruned / total

def replace_conv2d_with_compressed(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            compressed = CompressedConv2d(
                module.in_channels, module.out_channels, module.kernel_size[0],
                stride=module.stride[0], padding=module.padding[0],
                dilation=module.dilation[0], groups=module.groups,
                bias=(module.bias is not None))
            compressed.weight.data = module.weight.data.clone()
            if module.bias is not None:
                compressed.bias.data = module.bias.data.clone()
            setattr(model, name, compressed)
        elif len(list(module.children())) > 0:
            replace_conv2d_with_compressed(module)
    return model