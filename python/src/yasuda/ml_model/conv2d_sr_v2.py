from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.yasuda.ml_model.conv2d_block import DecoderBlock

logger = getLogger()


class ConvSrNetVer02(nn.Module):
    def __init__(
        self,
        *,
        n_encoder_blocks: int,
        feat_channels_0: int,
        feat_channels_1: int,
        feat_channels_2: int,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        bias: bool = False,
        **kwargs,
    ):
        super(ConvSrNetVer02, self).__init__()

        p = kernel_size // 2

        layers = []
        for i in range(n_encoder_blocks):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else feat_channels_0,
                    feat_channels_0,
                    kernel_size=kernel_size,
                    padding=p,
                    bias=bias,
                )
            )
            layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

        self.decoder = nn.Sequential(
            DecoderBlock(
                in_channels=feat_channels_0,
                out_channels=feat_channels_1,
                kernel_size=kernel_size,
                bias=bias,
                num_layers=2,
            ),
            DecoderBlock(
                in_channels=feat_channels_1,
                out_channels=feat_channels_2,
                kernel_size=kernel_size,
                bias=bias,
                num_layers=2,
            ),
            nn.Conv2d(
                feat_channels_2,
                out_channels,
                kernel_size=kernel_size,
                padding=p,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _x = F.interpolate(x, scale_factor=4, mode="nearest")

        y = self.encoder(x)
        y = self.decoder(y)

        return y + _x