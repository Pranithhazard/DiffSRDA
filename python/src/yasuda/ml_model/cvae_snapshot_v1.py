import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ref:
# https://github.com/unnir/cVAE/blob/baad46d96b78c4e604bb3c49ad3c898b797c0709/cvae.py


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, bias: bool = False):
        super(ResBlock, self).__init__()

        p = kernel_size // 2

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=p,
                bias=bias,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=p,
                bias=bias,
            ),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.convs(x))


class CVaeSnapshotVer01(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        encode_feat_channels: int = 256,
        n_encode_blocks: int = 5,
        decode_feat_channels: int = 32,
        n_decode_layers: int = 2,
        kernel_size: int = 3,
        bias: bool = False,
        lr_nx: int = 32,
        lr_ny: int = 16,
        hr_nx: int = 128,
        hr_ny: int = 64,
    ):
        super(CVaeSnapshotVer01, self).__init__()

        p = kernel_size // 2

        self.lr_nx = lr_nx
        self.lr_ny = lr_ny
        self.hr_nx = hr_nx
        self.hr_ny = hr_ny

        self.encode_feat_extractor = nn.Conv2d(
            in_channels * 2,
            encode_feat_channels,
            kernel_size=kernel_size,
            padding=p,
            bias=bias,
        )

        layers = []
        for _ in range(n_encode_blocks):
            layers.append(
                ResBlock(
                    in_channels=encode_feat_channels, kernel_size=kernel_size, bias=bias
                )
            )
        self.encoder = nn.Sequential(*layers)

        self.encoder_mu = nn.Conv2d(
            encode_feat_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=p,
            bias=True,
        )

        self.encoder_logvar = nn.Conv2d(
            encode_feat_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=p,
            bias=True,
        )

        self.decode_feat_extractor = nn.Conv2d(
            in_channels,
            decode_feat_channels,
            kernel_size=kernel_size,
            padding=p,
            bias=bias,
        )

        layers = []
        for _ in range(n_decode_layers):
            layers.append(
                nn.Conv2d(
                    decode_feat_channels,
                    decode_feat_channels,
                    kernel_size=kernel_size,
                    padding=p,
                    bias=bias,
                )
            )
            layers.append(nn.LeakyReLU())
        self.decoder = nn.Sequential(*layers)

        self.decoder_mu = nn.Conv2d(
            decode_feat_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=p,
            bias=True,
        )

    def encode(
        self, x: torch.Tensor, obs: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        # Check channel, x, y dims
        assert x.shape[1:] == (1, self.lr_nx, self.lr_ny)
        assert obs.shape[1:] == (1, self.hr_nx, self.hr_ny)

        _x = F.interpolate(x, size=(self.hr_nx, self.hr_ny), mode="nearest")

        f = torch.cat([_x, obs], dim=1)  # concat along channel dim
        f = self.encode_feat_extractor(f)

        y = self.encoder(f)
        y = y + f

        mu = self.encoder_mu(y)
        logvar = self.encoder_logvar(y)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        # Check channel, x, y dims
        assert z.shape[1:] == (1, self.hr_nx, self.hr_ny)

        f = self.decode_feat_extractor(z)
        y = self.decoder(f)
        y = y + f

        return self.decoder_mu(y)

    def forward(
        self, x: torch.Tensor, obs: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, obs)
        z = self.reparameterize(mu, logvar)
        return self.decode(x, z), mu, logvar