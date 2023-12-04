import torch
from torch import nn
from torch.nn import functional as F

from NetHelper import *
from UNetLayer import *

class UNet(nn.Module):
    def __init__(self,dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3, self_condition=False, resnet_block_groups=4):
        super().__init__()
        
        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1],dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

if __name__ == '__main__':
    pass