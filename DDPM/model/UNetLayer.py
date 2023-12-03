import torch
from torch import nn
from torch.nn import functional as F

from einops import reduce, rearrange
from numpy import einsum

from functools import partial
import math

from NetHelper import *


class PositionalEmbedding(nn.Module):
    def __init__(self,n_dim:int):
        super().__init__()

        assert n_dim % 2 == 0
        self.n_dim = n_dim

    def forward(self, t:torch.Tensor):
        device = t.device

        half_dim = self.n_dim // 2 # sin,cos 두개를 계산해서 concat하기 때문에 2로 나눠줌
        emb = math.log(10000) / (half_dim - 1)
        pos = torch.arange(half_dim,device=device)
        emb = torch.exp(pos * -emb)
        
        emb = t[:,None] * emb[None,:]
        emb = torch.cat((emb.sin(),emb.cos()),dim=-1)
        
        return emb
    
## ResNet block
class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    kernel weight를 normalization 시켜준다.
    3x3x3 kernel이 5개 있으면 각 5개에 대해 normalization을 해준다.
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1",partial(torch.var,unbiased=False))
        normalized_weight = (weight - mean) / (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim,dim_out,3,padding=1)
        self.norm = nn.GroupNorm(groups,dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        
        x = self.act(x)
        return x
    
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups=8):
        super().__init__()
        self.mlp(
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim,dim_out*2))
            if exists(time_emb_dim) else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out,1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
    
class Attention(nn.Module):
    def __init__(self,dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3,dim = 1)
        q,k,v = map(
            lambda t: rearrange(t,"b (h c) x y -> b h c (x y)", h = self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j",q,k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y",x=h, y=w)
        return self.to_out(out)
    
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim , 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k , v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)",h=self.heads),qkv
        )

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e",k,v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y",h=self.heads, x=h,y=w)
        return self.to_out(out)
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.nrom(x)
        return self.fn(x)

if __name__ == '__main__':
    b =torch.randn((3,8,100,100))
    a = torch. randn((3,8,100,50))
    c = einsum("b h d i, b h d j -> b h i j", a, b)
    print(c.shape)