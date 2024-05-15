import math
from inspect import isfunction
from functools import partial

from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

from .UnetLayer import *

class Unet(nn.Module):
    def __init__(
      self,
      dim,
      init_dim=None,
      out_dim=None,
      dim_mults=(1,2,4,8)  ,
      channels=3,
      self_condition=False,
      resnet_block_groups=4
    ):
        '''
        channels -> init_dim -> dim
        '''
        super().__init__()
        
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1) 
        
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)] # [init_dim, dim, dim*2, dim*4, dim*8]
        in_out = list(zip(dims[:-1], dims[1:])) # (init_dim, dim) (dim, dim*2) (dim*2, dim*4) (dim*4, dim*8)
        
        block_klass = partial(ResnetBlock, groups = resnet_block_groups)
        
        # time embeddings
        time_dim = dim * 4 # why dim * 4?
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in,LinearAttention(dim_in))),
                        Downsample(dim_in,dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3 , padding=1)
                    ]
                )
            )
        
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim,Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out)-1)
            
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )
            
        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
        
    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x),dim=1)
            
        x = self.init_conv(x) # b, c, x, y -> b, init_dim, x, y
        r = x.clone() # for last residual concat
        
        t = self.time_mlp(time) # (b,) -> (b, time_dim)
        
        h = []
        
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            
            x = downsample(x)
        
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            
            x = torch.cat((x, h.pop()),dim = 1)
            x = block2(x,t)
            x = attn(x)
            
            x = upsample(x)
            
        x = torch.cat((x, r), dim = 1)
        
        x = self.final_res_block(x, t)
        return self.final_conv(x)
    
if __name__ == "__main__":
    inp = torch.randn(10,3,128,128)
    t = torch.randint(low=0,high=100,size = (10,))
    print(inp.size(),t.size())
    unet = Unet(64,64,3)
    print(unet(inp,t).size())