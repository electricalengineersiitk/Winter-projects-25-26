# compressed linear layer with mask support for pruning
# basically nn.Linear but with a binary mask that zeros out pruned weights
# during forward pass: effective_weight = weight * mask
# the mask is a buffer not a parameter so it wont get gradients but it moves with the model to gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CompressedLinear(nn.Module):
    # linear layer that supports pruning masks
    # mask is all 1s initially (nothing pruned), then we zero out entries to prune
    def __init__(self,in_features,out_features,bias=True):
        super(CompressedLinear,self).__init__()
        self.in_features=in_features
        self.out_features=out_features

        # the actual learnable weight matrix and bias
        self.weight=nn.Parameter(torch.empty(out_features,in_features))
        if bias:
            self.bias=nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias',None)

        # mask starts as all ones = nothing pruned yet
        self.register_buffer('mask',torch.ones(out_features,in_features))

        # kaiming init same as default nn.Linear
        nn.init.kaiming_uniform_(self.weight,a=np.sqrt(5))
        if self.bias is not None:
            fan_in,_=nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound=1/np.sqrt(fan_in)
            nn.init.uniform_(self.bias,-bound,bound)

    def forward(self,x):
        # multiply weight by mask so pruned weights are zero
        masked_weight=self.weight*self.mask
        return F.linear(x,masked_weight,self.bias)

    def set_mask(self,mask):
        # update the mask (clone it so we dont mess up the original)
        self.mask.data=mask.clone().to(self.weight.device)

    def get_sparsity(self):
        # what fraction of weights are pruned (zeroed out)
        total=self.mask.numel()
        pruned=(self.mask==0).sum().item()
        return pruned/total

    def extra_repr(self):
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}, '
                f'sparsity={self.get_sparsity():.2%}')


def replace_linear_with_compressed(model):
    # swap all nn.Linear layers for CompressedLinear
    # this recursively searches the model so it works on any architecture
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            compressed = CompressedLinear(
                module.in_features, module.out_features,
                bias=(module.bias is not None))
            compressed.weight.data = module.weight.data.clone() # copy trained weights
            if module.bias is not None:
                compressed.bias.data = module.bias.data.clone()
            setattr(model, name, compressed)
        elif len(list(module.children())) > 0:
            # If it's a Sequential or container, dive inside it!
            replace_linear_with_compressed(module)

    return model
