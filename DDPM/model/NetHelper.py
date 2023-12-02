'''
Hugging Face의 코드 방식
network 구현에 도움을 주는 함수,클래스를 미리 만들어 놓음
'''
from inspect import isfunction

from einops.layers.torch import Rearrange
import torch
from torch import nn


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def num_to_group(num, divisor):
    '''
    group마다 item이 몇개씩 할당돼야 하는지 arr로 return
    '''
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class Residual(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kargs):
        return self.fn(x,*args,**kargs) + x
    
def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2,mode='nearest'),
        nn.Conv2d(dim, default(dim_out,dim), 3, padding=1)
    )

def Downsample(dim, dim_out=None):
    '''
    h와 w를 반으로 나누고 4배 채널을 늘려줌
    그 후 point conv로 채널 조정
    '''
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w",p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim),1)
    )

if __name__ == '__main__':
    a = torch.randn((3,3,100,100))

    