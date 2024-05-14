import math
from abc import abstractmethod

import torch
from torch import nn
import torch.nn.functional as F

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
        
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2: # if odd
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TimestepBlock(nn.Module):
    '''
    class for forward() takes timestep embedding
    '''

    @abstractmethod
    def forward(self, x, emb):
        """
        x & emb
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
                
        return x


### For 2D Convolution ###

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x,*args, **kwargs) + x
    
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv

        if self.use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)

    def forward(self, x):
        assert x.shape[1] == self.channels

        x = F.interpolate(x, scale_factor=2, mode="nearest")

        if self.use_conv:
            x = self.conv(x)
    
        return x
    
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.stride = 2

        if self.use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size = 3, stride = self.stride ,padding = 1)
        else:
            self.op = nn.AvgPool2d(kernel_size = self.stride)
        
    def forward(self, x):
        assert x.shape[1] == self.channels

        return self.op(x)
    
### For 2D Convolution ###

### Make UNet Block ##

class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm = False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32,self.channels),
            nn.SiLU(),
            nn.Conv2d(self.channels,self.out_channels,3,padding=1)
        )

        # for time embedding
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels,
                      2 * self.out_channels if use_scale_shift_norm else self.out_channels
            )
        )
        
        # out layer init zero for residual training
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32,self.out_channels),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            zero_module(nn.Conv2d(self.out_channels,self.out_channels,3,padding=1))
        )

        # align channel for residual connection

        if self.out_channels == self.channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(self.channels,self.out_channels,3,padding=1)
        else:
            self.skip_connection = nn.Conv2d(self.channels,self.out_channels,1,padding=0)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[...,None]
        
        # out_layer init zero
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:] # for GroupNorm -> scale & shift -> rest operation
            scale, shift = torch.chunk(emb_out,2,dim=1)
            h = out_norm(h)
            h = h * (1+scale) + shift #h + (h*scale + shift)
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h # h learn residual of x
### For Attention ###

class QKVAttention(nn.Module):
    def forward(self, qkv):
        channel = qkv.shape[1] // 3

        q,k,v = torch.split(qkv, channel, dim=1) # (batch, channel, w*h)
        scale = 1 / math.sqrt(math.sqrt(channel)) # https://github.com/openai/guided-diffusion/issues/38 why sqrt two times....

        attn_score = torch.einsum("bct,bcs -> bts", q*scale, k*scale) # Do (batch, w*h, channel) dot (batch, w*h, channel)^T = (b,t,c) dot (b,s,c)^T = (b,t,c) dot (b,c,s) = (b,t,s)
        attn_score = torch.softmax(attn_score.float(), dim = -1).type(attn_score.dtype)
        return torch.einsum("bts, bcs -> bct", attn_score, v) # (b,t,s) dot (b,s,c) = (b,t,c) -> align original shape = (b,c,t)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(32, self.channels)
        self.qkv = nn.Conv1d(self.channels,self.channels * 3, 1) 
        self.attention = QKVAttention()
        self.proj_out = zero_module(nn.Conv1d(self.channels,self.channels,1))

    def forward(self, x):
        b, c, *spatial = x.size()

        x = x.reshape(b,c,-1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2]) # channel dim split to num head

        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)

        return (x+h).reshape(b,c,*spatial)

### For Attention ###

### Make UNet Block ##

if __name__ == '__main__':
    a = Residual(nn.Linear(3,4))
    b = Upsample(3,True)
    c = ResBlock(64,10,0.1,128,True,True)
    inp = torch.randn(1,64, 10, 10)
    emb = torch.randn(64,10)
    d = AttentionBlock(64)
    d(inp)
    a = timestep_embedding(torch.arange(0,10),10)
    print(a.size())