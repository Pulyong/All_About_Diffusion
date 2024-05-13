import math

import torch
from torch import nn
import torch.nn.functional as F

from UnetLayer import *

class UNet(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks,attention_resolutions, dropout=0, channel_mult=(1,2,4,8), conv_resample=True, num_classes=None, num_heads=1, num_heads_upsample=-1, use_scale_shift_norm = False):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))
            ]
        )

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        mult * model_channels,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ]

                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm
            )
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels*mult,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32,ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels,out_channels,3,padding=1))
        )

    def forward(self, x, timesteps, y=None):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"        

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps,self.model_channels))
        

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()],dim = 1)
            h = module(cat_in, emb)
        
        return self.out(h)
    
if __name__ == '__main__':
    unet = UNet(3,64,3,2,[16,8]).to('cuda:0')
    inp = torch.randn(3,3,256,256).to('cuda:0')
    emb = torch.arange(0,3).to('cuda:0')
