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


class CVaeSnapshotVer03(nn.Module):
    def __init__(
        self,
        *,
        encode_feat_channels: int,
        n_feat_blocks: int,
        n_encode_blocks: int,
        decode_feat_channels: int,
        n_decode_layers: int,
        is_skipped_globally_encoder: bool,
        is_skipped_globally_decoder: bool,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        bias: bool = False,
        lr_nx: int = 32,
        lr_ny: int = 16,
        hr_nx: int = 128,
        hr_ny: int = 64,
        **kwargs,
    ):
        super(CVaeSnapshotVer03, self).__init__()

        assert n_feat_blocks > 1 and n_encode_blocks > 1
        assert n_decode_layers != 1

        p = kernel_size // 2

        self.lr_nx = lr_nx
        self.lr_ny = lr_ny
        self.hr_nx = hr_nx
        self.hr_ny = hr_ny
        self.n_decode_layers = n_decode_layers
        self.is_skipped_globally_encoder = is_skipped_globally_encoder
        self.is_skipped_globally_decoder = is_skipped_globally_decoder

        layers = []
        for i in range(n_feat_blocks):
            layers.append(
                nn.Conv2d(
                    2 * in_channels if i == 0 else encode_feat_channels,
                    encode_feat_channels,
                    kernel_size=kernel_size,
                    padding=p,
                    bias=bias,
                )
            )
            layers.append(nn.ReLU())
        self.encode_x_feat_extractor = nn.Sequential(*layers)

        layers = []
        for i in range(n_feat_blocks):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else encode_feat_channels,
                    encode_feat_channels,
                    kernel_size=kernel_size,
                    padding=p,
                    bias=bias,
                )
            )
            layers.append(nn.ReLU())
        self.encode_o_feat_extractor = nn.Sequential(*layers)

        layers = []
        for i in range(n_encode_blocks):
            layers.append(
                ResBlock(
                    2 * encode_feat_channels,
                    kernel_size=kernel_size,
                    bias=bias,
                )
            )
        layers.append(
            nn.Conv2d(
                2 * encode_feat_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=p,
                bias=True,
            )
        )
        self.encoder_mu = nn.Sequential(*layers)

        layers = []
        for i in range(n_encode_blocks):
            layers.append(
                ResBlock(
                    2 * encode_feat_channels,
                    kernel_size=kernel_size,
                    bias=bias,
                )
            )
        layers.append(
            nn.Conv2d(
                2 * encode_feat_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=p,
                bias=True,
            )
        )
        self.encoder_logvar = nn.Sequential(*layers)

        if self.n_decode_layers > 0:
            layers = []
            for i in range(self.n_decode_layers - 1):
                layers.append(
                    nn.Conv2d(
                        in_channels if i == 0 else decode_feat_channels,
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
        assert x.shape[1:] == (1, self.lr_ny, self.lr_nx)
        assert obs.shape[1:] == (1, self.hr_ny, self.hr_nx)

        _x = F.interpolate(x, size=(self.hr_ny, self.hr_nx), mode="nearest")

        fx = torch.cat([_x, obs], dim=1)  # concat along channel dim
        fx = self.encode_x_feat_extractor(fx)
        fo = self.encode_o_feat_extractor(obs)

        y = torch.cat([fx, fo], dim=1)  # concat along channel dim

        logvar = self.encoder_logvar(y)
        mu = self.encoder_mu(y)

        if self.is_skipped_globally_encoder:
            mu = mu + _x

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:

        # Check channel, x, y dims
        assert z.shape[1:] == (1, self.hr_ny, self.hr_nx)

        if self.n_decode_layers == 0:
            return z
        else:
            y = self.decoder(z)
            y = self.decoder_mu(y)

            if self.is_skipped_globally_decoder:
                y = y + z

            return y

    def forward(
        self, x: torch.Tensor, obs: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, obs)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar