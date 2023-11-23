import torch
from torch import nn
from torch.nn import functional as F

class TimeEmbedding(nn.Module):
    def __init__(self,n_channles: int):
        super().__init__()
        self.n_channels = n_channles

        self.act = nn.SiLU()