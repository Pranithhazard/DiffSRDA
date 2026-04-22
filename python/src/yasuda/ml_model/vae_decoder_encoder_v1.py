import typing

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int,
        bias: bool,
        type_down_sample: typing.Literal["conv", "average"],
        num_layers: int,
        negative_slope: float,
    ):
        super(EncoderBlock, self).__init__()

        assert num_layers >= 2

        p = kernel_size // 2

        if type_down_sample == "conv":
            self.down = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=p,
                    bias=bias,
                ),
                nn.LeakyReLU(negative_slope=negative_slope),
            )
        elif type_down_sample == "average":
            self.down = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=p,
                    bias=bias,
                ),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0),
            )
        else:
            raise Exception(f"Downsampling type of {type_down_sample} is not supported")

        convs = []
        for _ in range(num_layers - 1):
            convs.append(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=p,
                    bias=bias,
                )
            )
            convs.append(nn.LeakyReLU(negative_slope=negative_slope))

        self.convs = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        return self.convs(y)


class PixelShuffleBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        kernel_size: int,
        bias: bool,
        negative_slope: float,
        upscale_factor: int = 2,
    ):
        super(PixelShuffleBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            (upscale_factor**2) * in_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

        self.act = nn.LeakyReLU(negative_slope=negative_slope)

        self.upsample = nn.PixelShuffle(upscale_factor=upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.act(y)
        return self.upsample(y)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool,
        type_up_sample: typing.Literal["pixel_shuffle"],
        num_layers: int,
        negative_slope: float,
        upscale_factor: int = 2,
    ):
        super(DecoderBlock, self).__init__()

        assert num_layers >= 2

        if type_up_sample == "pixel_shuffle":
            self.up = PixelShuffleBlock(
                in_channels=in_channels,
                kernel_size=kernel_size,
                bias=bias,
                negative_slope=negative_slope,
                upscale_factor=upscale_factor,
            )
        else:
            raise Exception(f"Upsampling type of {type_up_sample} is not supported")

        convs = []
        for i in range(num_layers - 1):
            convs.append(
                nn.Conv2d(
                    (in_channels if i == 0 else out_channels),
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    bias=bias,
                )
            )
            convs.append(nn.LeakyReLU(negative_slope=negative_slope))

        self.convs = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.up(x)
        return self.convs(y)


class PoolingBlockToHalfSize(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        num_in_features: int,
        num_out_features: int,
        num_layers: int,
        bias: bool,
        negative_slope: float,
        pooling_method: typing.Literal["conv", "average"],
    ):
        super().__init__()

        assert num_layers > 1

        p = kernel_size // 2

        layers = []
        for _ in range(num_layers - 1):
            layers.append(
                nn.Conv2d(
                    num_in_features, num_in_features, kernel_size, bias=bias, padding=p
                )
            )
            layers.append(nn.LeakyReLU(negative_slope))

        if pooling_method == "average":
            layers.append(
                nn.Conv2d(
                    num_in_features, num_out_features, kernel_size, bias=bias, padding=p
                )
            )
            layers.append(nn.LeakyReLU(negative_slope))
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        elif pooling_method == "conv":
            layers.append(
                nn.Conv2d(
                    num_in_features,
                    num_out_features,
                    kernel_size,
                    stride=2,
                    bias=bias,
                    padding=p,
                )
            )
            layers.append(nn.LeakyReLU(negative_slope))
        else:
            raise NotImplementedError(f"{pooling_method} is not supported")

        self.pooling_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooling_block(x)


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feat_channels: int,
        kernel_size: int,
        bias: bool,
        type_down_sample: typing.Literal["conv", "average"],
        num_layers_in_encoding_block: int,
        num_layers_in_latent_mapper: int,
        num_layers_in_decoding_block: int,
        negative_slope: float,
    ):
        super().__init__()

        scale_factor = 2

        self.down0 = EncoderBlock(
            in_channels=in_channels,
            out_channels=feat_channels,
            stride=scale_factor,
            kernel_size=kernel_size,
            bias=bias,
            type_down_sample=type_down_sample,
            num_layers=num_layers_in_encoding_block,
            negative_slope=negative_slope,
        )

        self.down1 = EncoderBlock(
            in_channels=feat_channels,
            out_channels=feat_channels,
            stride=scale_factor,
            kernel_size=kernel_size,
            bias=bias,
            type_down_sample=type_down_sample,
            num_layers=num_layers_in_encoding_block,
            negative_slope=negative_slope,
        )

        layers = []
        for _ in range(num_layers_in_latent_mapper):
            layers.append(
                nn.Conv2d(
                    in_channels=feat_channels,
                    out_channels=feat_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=bias,
                )
            )
            layers.append(nn.LeakyReLU(negative_slope))

        self.mapper = nn.Sequential(*layers)

        self.up1 = DecoderBlock(
            in_channels=feat_channels + feat_channels,
            out_channels=feat_channels,
            kernel_size=kernel_size,
            bias=bias,
            num_layers=num_layers_in_decoding_block,
            negative_slope=negative_slope,
            upscale_factor=scale_factor,
            type_up_sample="pixel_shuffle",
        )

        self.up0 = DecoderBlock(
            in_channels=feat_channels + feat_channels,
            out_channels=feat_channels,
            kernel_size=kernel_size,
            bias=bias,
            num_layers=num_layers_in_decoding_block,
            negative_slope=negative_slope,
            upscale_factor=scale_factor,
            type_up_sample="pixel_shuffle",
        )

        self.reconst = nn.Conv2d(
            in_channels=feat_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            bias=bias,
            padding=kernel_size // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.down0(x)
        x1 = self.down1(x0)

        z = self.mapper(x1)

        y1 = self.up1(torch.cat([z, x1], dim=1))
        y0 = self.up0(torch.cat([y1, x0], dim=1))

        y = self.reconst(y0)

        return y


class VaeDecoderVer01(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool,
        negative_slope: float,
        decoder_feat_layers: int,
        decoder_num_features0: int,
        decoder_num_features1: int,
        decoder_num_features2: int,
        decoder_num_layers_in_block: int,
        decoder_upsampling_method: str,
        has_decoder_global_skip_connection: bool,
        **kwargs,
    ):
        super().__init__()

        p = kernel_size // 2
        scale_factor_per_block = 2
        self.decoder_upsampling_method = decoder_upsampling_method
        self.has_decoder_global_skip_connection = has_decoder_global_skip_connection

        layers = []
        for i in range(decoder_feat_layers):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else decoder_num_features0,
                    decoder_num_features0,
                    kernel_size,
                    padding=p,
                    bias=bias,
                )
            )
            layers.append(nn.LeakyReLU(negative_slope))
        self.decoder_feat_extractor = nn.Sequential(*layers)

        self.decoder_reconstructor = nn.Sequential(
            DecoderBlock(
                in_channels=decoder_num_features0,
                out_channels=decoder_num_features1,
                kernel_size=kernel_size,
                bias=bias,
                num_layers=decoder_num_layers_in_block,
                negative_slope=negative_slope,
                upscale_factor=scale_factor_per_block,
                type_up_sample="pixel_shuffle",
            ),
            DecoderBlock(
                in_channels=decoder_num_features1,
                out_channels=decoder_num_features2,
                kernel_size=kernel_size,
                bias=bias,
                num_layers=decoder_num_layers_in_block,
                negative_slope=negative_slope,
                upscale_factor=scale_factor_per_block,
                type_up_sample="pixel_shuffle",
            ),
            nn.Conv2d(
                decoder_num_features2,
                out_channels,
                kernel_size=kernel_size,
                padding=p,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        f = self.decoder_feat_extractor(x)
        y = self.decoder_reconstructor(f)

        if not self.has_decoder_global_skip_connection:
            return y

        _x = F.interpolate(x, scale_factor=4, mode=self.decoder_upsampling_method)

        return _x + y


class VaeEncoderVer01(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_obs_features0: int,
        num_obs_features1: int,
        num_obs_features2: int,
        num_obs_layers0: int,
        num_obs_layers1: int,
        num_lr_features: int,
        num_lr_layers: int,
        encoder_num_features0: int,
        encoder_num_features1: int,
        encoder_down_sampling_method: typing.Literal["conv", "average"],
        encoder_num_layers_in_encoding_block: int,
        encoder_num_layers_in_latent_mapper: int,
        encoder_num_layers_in_decoding_block: int,
        has_encoder_global_skip_connection: bool,
        bias: bool,
        negative_slope: float,
        pooling_method: typing.Literal["conv", "average"],
        **kwargs,
    ):
        super().__init__()

        self.has_encoder_global_skip_connection = has_encoder_global_skip_connection
        p = kernel_size // 2

        layers = []
        layers.append(
            nn.Conv2d(
                in_channels, num_obs_features0, kernel_size, padding=p, bias=False
            )
        )
        layers.append(
            PoolingBlockToHalfSize(
                kernel_size=kernel_size,
                num_in_features=num_obs_features0,
                num_out_features=num_obs_features1,
                num_layers=num_obs_layers0,
                bias=bias,
                negative_slope=negative_slope,
                pooling_method=pooling_method,
            )
        )
        layers.append(
            PoolingBlockToHalfSize(
                kernel_size=kernel_size,
                num_in_features=num_obs_features1,
                num_out_features=num_obs_features2,
                num_layers=num_obs_layers1,
                bias=bias,
                negative_slope=negative_slope,
                pooling_method=pooling_method,
            )
        )
        self.obs_feat_extractor = nn.Sequential(*layers)

        layers = []
        layers.append(
            nn.Conv2d(in_channels, num_lr_features, kernel_size, padding=p, bias=False)
        )
        for _ in range(num_lr_layers):
            layers.append(
                nn.Conv2d(
                    num_lr_features, num_lr_features, kernel_size, padding=p, bias=bias
                )
            )
            layers.append(nn.LeakyReLU(negative_slope))

        self.lr_feat_extractor = nn.Sequential(*layers)

        self.bottle_neck = nn.Conv2d(
            num_lr_features + num_obs_features2,
            encoder_num_features0,
            kernel_size=1,
            padding=0,
            bias=True,
        )

        self.unet = Unet(
            in_channels=encoder_num_features0,
            feat_channels=encoder_num_features1,
            kernel_size=kernel_size,
            bias=bias,
            type_down_sample=encoder_down_sampling_method,
            num_layers_in_encoding_block=encoder_num_layers_in_encoding_block,
            num_layers_in_latent_mapper=encoder_num_layers_in_latent_mapper,
            num_layers_in_decoding_block=encoder_num_layers_in_decoding_block,
            negative_slope=negative_slope,
        )

        last_num_features = encoder_num_features0
        if self.has_encoder_global_skip_connection:
            last_num_features += encoder_num_features0

        self.reconstruct_mu = nn.Conv2d(
            last_num_features, out_channels, kernel_size, padding=p, bias=True
        )

        self.reconstruct_logvar = nn.Conv2d(
            last_num_features, out_channels, kernel_size, padding=p, bias=True
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor, obs: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        feat_obs = self.obs_feat_extractor(obs)
        feat_x = self.lr_feat_extractor(x)

        # Concatenate along channel dim
        feat0 = self.bottle_neck(torch.cat([feat_obs, feat_x], dim=1))
        feat1 = self.unet(feat0)

        if self.has_encoder_global_skip_connection:
            feat1 = torch.cat([feat1, feat0], dim=1)

        mu = self.reconstruct_mu(feat1)
        logvar = self.reconstruct_logvar(feat1)

        return mu, logvar