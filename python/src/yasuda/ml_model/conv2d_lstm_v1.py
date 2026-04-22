import typing
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
        in_channels: int,
        x_feat_channels_0: int,
        x_feat_channels_1: int,
        x_feat_channels_2: int,
        o_feat_channels_0: int,
        o_feat_channels_1: int,
        o_feat_channels_2: int,
        o_feat_channels_3: int,
        out_channels: int,
        kernel_size: int = 3,
        num_layers_x_encoder: int = 2,
        num_layers_o_encoder: int = 2,
        bias_x_encoder: bool = False,
        bias_o_encoder: bool = False,
        scale_factor: int = 4,
    ):
        super(Encoder, self).__init__()
        self.scale_factor = scale_factor

        self.x_encoder = nn.Sequential(
            EncoderBlock(
                in_channels=(x_feat_channels_0 + o_feat_channels_0),
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

        self.o_encoder = nn.Sequential(
            EncoderBlock(
                in_channels=o_feat_channels_0,
                out_channels=o_feat_channels_1,
                kernel_size=kernel_size,
                stride=2,
                bias=bias_o_encoder,
                num_layers=num_layers_o_encoder,
            ),
            EncoderBlock(
                in_channels=o_feat_channels_1,
                out_channels=o_feat_channels_2,
                kernel_size=kernel_size,
                stride=2,
                bias=bias_o_encoder,
                num_layers=num_layers_o_encoder,
            ),
            EncoderBlock(
                in_channels=o_feat_channels_2,
                out_channels=o_feat_channels_3,
                kernel_size=kernel_size,
                stride=2,
                bias=bias_o_encoder,
                num_layers=num_layers_o_encoder,
            ),
            EncoderBlock(
                in_channels=o_feat_channels_3,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                bias=bias_o_encoder,
                num_layers=num_layers_o_encoder,
            ),
        )

    def forward(
        self, x: torch.Tensor, obs: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        y = torch.cat([x, obs], dim=1)  # concat along channel dims
        y = self.x_encoder(y)

        z = self.o_encoder(obs)

        return y, z


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


class OptInterpWeightCalculator(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super(OptInterpWeightCalculator, self).__init__()

        self.weight_calculator = nn.Sequential(
            nn.Linear(input_size, output_size, bias=bias),
            nn.ReLU(),
            nn.Linear(output_size, output_size, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight_calculator(x)


class LstmTimeSeriesMappingBlock(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        sequence_length: int,
        bias: bool = False,
        bidirectional: bool = False,
        skip_lstm: bool = False,
    ):
        super(LstmTimeSeriesMappingBlock, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.bidirectional = bidirectional
        self.skip_lstm = skip_lstm

        self.w_calc = OptInterpWeightCalculator(
            input_size=2 * input_size,
            output_size=input_size,
            bias=bias,
        )

        if not self.skip_lstm:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=output_size,
                num_layers=1,
                bias=bias,
                batch_first=True,
                bidirectional=bidirectional,
            )

            if bidirectional:
                # batch, time, and latent dims
                w = torch.full(size=(1, sequence_length, 1), fill_value=0.5)
                self.lstm_w = nn.Parameter(data=w, requires_grad=True)

    def forward(self, x: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        _x = x.contiguous().view(-1, self.input_size)
        _o = obs.view(-1, self.input_size)

        w = self.w_calc(torch.cat([_x, _o], dim=1))
        w = w.view(-1, self.sequence_length, self.input_size)

        y1 = w * x + (1.0 - w) * obs
        if self.skip_lstm:
            return y1

        y1, _ = self.lstm(y1)

        if not self.bidirectional:
            return y1

        # Split out between the forward and backward directions
        y1 = y1.view(-1, self.sequence_length, 2, self.output_size)

        y1 = self.lstm_w * y1[:, :, 0, :] + (1.0 - self.lstm_w) * y1[:, :, 1, :]

        return y1


class TimeSeriesMapper(nn.Module):
    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        n_lstm_blocks: int = 1,
        bias: bool = False,
        bidirectional: bool = False,
        skip_lstm: bool = False,
    ):
        super(TimeSeriesMapper, self).__init__()

        self.skip_lstm = skip_lstm

        self.lstms = nn.Sequential(
            *[
                LstmTimeSeriesMappingBlock(
                    input_size=input_size,
                    output_size=input_size,
                    sequence_length=sequence_length,
                    bias=bias,
                    bidirectional=bidirectional,
                    skip_lstm=skip_lstm,
                )
                for _ in range(n_lstm_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        y = x
        for lstm in self.lstms:
            y = lstm(y, obs)

        if self.skip_lstm:
            return y

        return y + x  # skip connection


class ConvLstmSrDaNetVer01(nn.Module):
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
        sequence_length: int,
        bidirectional: bool = False,
        skip_lstm: bool = False,
        n_lstm_blocks: int = 1,
        scale_factor: int = 4,
        lr_x_size: int = 32,
        lr_y_size: int = 16,
        kernel_size: int = 3,
        bias: bool = False,
    ):
        super(ConvLstmSrDaNetVer01, self).__init__()

        p = kernel_size // 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sequence_length = sequence_length
        logger.info(f"Sequence length = {self.sequence_length}")

        self.scale_factor = scale_factor
        self.lr_x_size = lr_x_size
        self.lr_y_size = lr_y_size
        self.hr_x_size = lr_x_size * self.scale_factor
        self.hr_y_size = lr_y_size * self.scale_factor

        # 16 == 2**4 (encoder has 4 blocks to down sample)
        self.latent_x_size = self.hr_x_size // 16
        self.latent_y_size = self.hr_y_size // 16

        self.latent_channels = latent_channels
        self.latent_dim = self.latent_x_size * self.latent_y_size * self.latent_channels

        logger.info(f"LR size y = {self.lr_y_size}, x = {self.lr_x_size}")
        logger.info(f"HR size y = {self.hr_y_size}, x = {self.hr_x_size}")
        logger.info(f"Latent size y = {self.latent_y_size}, x = {self.latent_x_size}")
        logger.info(
            f"latent dim = {self.latent_dim}, latent channels = {self.latent_channels}"
        )

        self.x_feat_extractor = nn.Conv2d(
            in_channels, feat_channels_0, kernel_size, padding=p, bias=bias
        )
        self.o_feat_extractor = nn.Conv2d(
            in_channels, feat_channels_0, kernel_size, padding=p, bias=bias
        )
        self.encoder = Encoder(
            in_channels=3,
            x_feat_channels_0=feat_channels_0,
            x_feat_channels_1=feat_channels_1,
            x_feat_channels_2=feat_channels_2,
            o_feat_channels_0=feat_channels_0,
            o_feat_channels_1=feat_channels_0,
            o_feat_channels_2=feat_channels_0,
            o_feat_channels_3=feat_channels_0,
            out_channels=latent_channels,
        )

        self.ts_mapper = TimeSeriesMapper(
            input_size=self.latent_dim,
            sequence_length=self.sequence_length,
            n_lstm_blocks=n_lstm_blocks,
            bias=bias,
            bidirectional=bidirectional,
            skip_lstm=skip_lstm,
        )

        self.decoder = Decoder(
            feat_channels_0=latent_channels,
            feat_channels_1=feat_channels_3,
            feat_channels_2=feat_channels_2,
            feat_channels_3=feat_channels_1,
            out_channels=feat_channels_0,
        )

        self.reconstructor = nn.Sequential(
            nn.Conv2d(
                3 * feat_channels_0, out_channels, kernel_size, padding=p, bias=True
            )
        )

    def forward(self, xs: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        assert xs.shape[2:] == (self.in_channels, self.lr_y_size, self.lr_x_size)
        assert obs.shape[2:] == (self.in_channels, self.hr_y_size, self.hr_x_size)

        x = xs.view(-1, self.in_channels, self.lr_y_size, self.lr_x_size)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")

        o = obs.view(-1, self.in_channels, self.hr_y_size, self.hr_x_size)

        feat_x = self.x_feat_extractor(x)
        feat_o = self.o_feat_extractor(o)

        latent_x, latent_o = self.encoder(feat_x, feat_o)

        latent_x = latent_x.view(-1, self.sequence_length, self.latent_dim)
        latent_o = latent_o.view(-1, self.sequence_length, self.latent_dim)

        latent_f = self.ts_mapper(latent_x, latent_o)
        latent_f = latent_f.contiguous().view(
            -1,
            self.latent_channels,
            self.latent_y_size,
            self.latent_x_size,
        )

        y = self.decoder(latent_f)
        y = torch.cat([y, feat_x, feat_o], dim=1)  # concat along channel
        y = self.reconstructor(y)

        return y.view(
            -1, self.sequence_length, self.out_channels, self.hr_y_size, self.hr_x_size
        )
