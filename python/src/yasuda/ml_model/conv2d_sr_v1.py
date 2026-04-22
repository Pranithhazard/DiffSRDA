from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.yasuda.ml_model.conv2d_block import DecoderBlock, EncoderBlock

logger = getLogger()


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        x_feat_channels_0: int,
        x_feat_channels_1: int,
        x_feat_channels_2: int,
        out_channels: int,
        kernel_size: int = 3,
        num_layers_x_encoder: int = 2,
        bias_x_encoder: bool = False,
        scale_factor: int = 4,
    ):
        super(Encoder, self).__init__()
        self.scale_factor = scale_factor

        self.x_encoder = nn.Sequential(
            EncoderBlock(
                in_channels=x_feat_channels_0,
                out_channels=x_feat_channels_1,
                kernel_size=kernel_size,
                stride=4,
                bias=bias_x_encoder,
                num_layers=num_layers_x_encoder,
            ),
            EncoderBlock(
                in_channels=x_feat_channels_1,
                out_channels=x_feat_channels_2,
                kernel_size=kernel_size,
                stride=2,
                bias=bias_x_encoder,
                num_layers=num_layers_x_encoder,
            ),
            EncoderBlock(
                in_channels=x_feat_channels_2,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                bias=bias_x_encoder,
                num_layers=num_layers_x_encoder,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.x_encoder(x)


class Decoder(nn.Module):
    def __init__(
        self,
        feat_channels_0: int,
        feat_channels_1: int,
        feat_channels_2: int,
        feat_channels_3: int,
        out_channels: int,
        kernel_size: int = 3,
        num_layers: int = 2,
        bias: bool = False,
    ):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            DecoderBlock(
                in_channels=feat_channels_0,
                out_channels=feat_channels_1,
                kernel_size=kernel_size,
                bias=bias,
                num_layers=num_layers,
            ),
            DecoderBlock(
                in_channels=feat_channels_1,
                out_channels=feat_channels_2,
                kernel_size=kernel_size,
                bias=bias,
                num_layers=num_layers,
            ),
            DecoderBlock(
                in_channels=feat_channels_2,
                out_channels=feat_channels_3,
                kernel_size=kernel_size,
                bias=bias,
                num_layers=num_layers,
            ),
            DecoderBlock(
                in_channels=feat_channels_3,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
                num_layers=num_layers,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class ConvSrNetVer01(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        feat_channels_0: int,
        feat_channels_1: int,
        feat_channels_2: int,
        feat_channels_3: int,
        latent_channels: int,
        out_channels: int,
        scale_factor: int = 4,
        lr_x_size: int = 32,
        lr_y_size: int = 16,
        kernel_size: int = 3,
        bias: bool = False,
    ):
        super(ConvSrNetVer01, self).__init__()

        p = kernel_size // 2

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.scale_factor = scale_factor
        self.lr_x_size = lr_x_size
        self.lr_y_size = lr_y_size
        self.hr_x_size = lr_x_size * self.scale_factor
        self.hr_y_size = lr_y_size * self.scale_factor

        # 16 == 2**4 (encoder has 4 blocks to down sample)
        self.latent_x_size = self.hr_x_size // 16
        self.latent_y_size = self.hr_y_size // 16

        self.latent_channels = latent_channels

        logger.info(f"LR size y = {self.lr_y_size}, x = {self.lr_x_size}")
        logger.info(f"HR size y = {self.hr_y_size}, x = {self.hr_x_size}")
        logger.info(f"Latent size y = {self.latent_y_size}, x = {self.latent_x_size}")
        logger.info(f"latent channels = {self.latent_channels}")
        logger.info(f"bias = {bias}")

        self.x_feat_extractor = nn.Conv2d(
            in_channels, feat_channels_0, kernel_size, padding=p, bias=bias
        )
        self.encoder = Encoder(
            x_feat_channels_0=feat_channels_0,
            x_feat_channels_1=feat_channels_1,
            x_feat_channels_2=feat_channels_2,
            out_channels=latent_channels,
            bias_x_encoder=bias,
        )

        self.opt_itp = nn.Sequential(
            nn.Conv2d(
                latent_channels,
                latent_channels,
                kernel_size=kernel_size,
                padding=p,
                bias=bias,
            ),
            nn.ReLU(),
            nn.Conv2d(
                latent_channels,
                latent_channels,
                kernel_size=kernel_size,
                padding=p,
                bias=bias,
            ),
        )

        self.decoder = Decoder(
            feat_channels_0=latent_channels,
            feat_channels_1=feat_channels_3,
            feat_channels_2=feat_channels_2,
            feat_channels_3=feat_channels_1,
            out_channels=feat_channels_0,
            bias=bias,
        )

        self.reconstructor = nn.Sequential(
            nn.Conv2d(
                2 * feat_channels_0, out_channels, kernel_size, padding=p, bias=True
            )
        )

    def forward(self, xs: torch.Tensor, obs: torch.Tensor = None) -> torch.Tensor:

        assert xs.shape[1:] == (
            self.in_channels,
            self.lr_y_size,
            self.lr_x_size,
        )

        x = F.interpolate(xs, scale_factor=self.scale_factor, mode="nearest")

        feat_x = self.x_feat_extractor(x)
        latent_x = self.encoder(feat_x)

        latent_f = self.opt_itp(latent_x)
        latent_f = latent_f + latent_x  # skip connection

        y = self.decoder(latent_f)
        y = torch.cat([y, feat_x], dim=1)  # concat along channel
        y = self.reconstructor(y)

        return y