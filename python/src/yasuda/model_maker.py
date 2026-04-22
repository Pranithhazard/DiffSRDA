import typing
from logging import getLogger

import torch
from src.yasuda.ml_model.conv2d_lstm_v1 import ConvLstmSrDaNetVer01
from src.yasuda.ml_model.conv2d_sr_v1 import ConvSrNetVer01
from src.yasuda.ml_model.conv2d_sr_v2 import ConvSrNetVer02
from src.yasuda.ml_model.conv2d_transformer_v1 import ConvTransformerSrDaNetVer01
from src.yasuda.ml_model.conv2d_transformer_v2 import ConvTransformerSrDaNetVer02
from src.yasuda.ml_model.conv2d_transformer_v3 import ConvTransformerSrDaNetVer03
from src.yasuda.ml_model.cvae_snapshot_v2 import CVaeSnapshotVer02
from src.yasuda.ml_model.cvae_snapshot_v3 import CVaeSnapshotVer03
from src.yasuda.ml_model.cvae_snapshot_v4 import CVaeSnapshotVer04
from src.yasuda.ml_model.interpolator import Interpolator
from src.yasuda.ml_model.vae_decoder_encoder_v1 import VaeDecoderVer01, VaeEncoderVer01

logger = getLogger()


def make_model(config: dict) -> torch.nn.Module:

    if config["model"]["model_name"] == "ConvLstmSrDaNetVer01":
        logger.info("ConvLstmSrDaNetVer01 is created")
        return ConvLstmSrDaNetVer01(
            in_channels=config["model"]["in_channels"],
            feat_channels_0=config["model"]["feat_channels_0"],
            feat_channels_1=config["model"]["feat_channels_1"],
            feat_channels_2=config["model"]["feat_channels_2"],
            feat_channels_3=config["model"]["feat_channels_3"],
            latent_channels=config["model"]["latent_channels"],
            out_channels=config["model"]["out_channels"],
            sequence_length=config["model"]["sequence_length"],
            bidirectional=config["model"]["bidirectional"],
            skip_lstm=config["model"]["skip_lstm"],
            n_lstm_blocks=config["model"]["n_lstm_blocks"],
        )
    elif config["model"]["model_name"] == "ConvTransformerSrDaNetVer01":
        logger.info("ConvTransformerSrDaNetVer01 is created")
        return ConvTransformerSrDaNetVer01(
            in_channels=config["model"]["in_channels"],
            feat_channels_0=config["model"]["feat_channels_0"],
            feat_channels_1=config["model"]["feat_channels_1"],
            feat_channels_2=config["model"]["feat_channels_2"],
            feat_channels_3=config["model"]["feat_channels_3"],
            latent_channels=config["model"]["latent_channels"],
            out_channels=config["model"]["out_channels"],
            n_multi_attention_heads=config["model"]["n_multi_attention_heads"],
            sequence_length=config["model"]["sequence_length"],
            n_transformer_blocks=config["model"]["n_transformer_blocks"],
            use_global_skip_connection_in_ts_mapper=config["model"][
                "use_global_skip_connection_in_ts_mapper"
            ],
            bias=config["model"]["bias"],
        )
    elif config["model"]["model_name"] == "ConvSrNetVer01":
        logger.info("ConvSrNetVer01 is created.")
        return ConvSrNetVer01(
            in_channels=config["model"]["in_channels"],
            feat_channels_0=config["model"]["feat_channels_0"],
            feat_channels_1=config["model"]["feat_channels_1"],
            feat_channels_2=config["model"]["feat_channels_2"],
            feat_channels_3=config["model"]["feat_channels_3"],
            latent_channels=config["model"]["latent_channels"],
            out_channels=config["model"]["out_channels"],
            bias=config["model"]["bias"],
        )
    elif config["model"]["model_name"] == "ConvTransformerSrDaNetVer02":
        logger.info("ConvTransformerSrDaNetVer02 is created")

        if "input_sampling_interval" in config["data"]:
            _interval = config["data"]["input_sampling_interval"]
        else:
            _interval = config["data"]["lr_input_sampling_interval"]

        return ConvTransformerSrDaNetVer02(
            in_channels=config["model"]["in_channels"],
            feat_channels_0=config["model"]["feat_channels_0"],
            feat_channels_1=config["model"]["feat_channels_1"],
            feat_channels_2=config["model"]["feat_channels_2"],
            feat_channels_3=config["model"]["feat_channels_3"],
            latent_channels=config["model"]["latent_channels"],
            out_channels=config["model"]["out_channels"],
            n_multi_attention_heads=config["model"]["n_multi_attention_heads"],
            sequence_length=config["model"]["sequence_length"],
            n_transformer_blocks=config["model"]["n_transformer_blocks"],
            use_global_skip_connection_in_ts_mapper=config["model"][
                "use_global_skip_connection_in_ts_mapper"
            ],
            bias=config["model"]["bias"],
            input_sampling_interval=_interval,
        )
    elif config["model"]["model_name"] == "CVaeSnapshotVer02":
        logger.info("CVaeSnapshotVer02 is created")
        return CVaeSnapshotVer02(
            n_encode_blocks=config["model"]["n_encode_blocks"],
            n_decode_layers=config["model"]["n_decode_layers"],
        )
    elif config["model"]["model_name"] == "ConvTransformerSrDaNetVer03":
        logger.info("ConvTransformerSrDaNetVer03 is created")

        if "input_sampling_interval" in config["data"]:
            _interval = config["data"]["input_sampling_interval"]
        elif "lr_time_interval" in config["data"]:
            _interval = config["data"]["lr_time_interval"]
        else:
            _interval = config["data"]["lr_input_sampling_interval"]

        return ConvTransformerSrDaNetVer03(
            input_sampling_interval=_interval, **config["model"]
        )
    else:
        raise NotImplementedError(f"{config['model']['model_name']} is not supported")


def make_prior_model(config: dict) -> torch.nn.Module:

    name = config["model"]["prior_model"]["name"]

    if name == "ConvSrNetVer02":
        logger.info("ConvSrNetVer02 is created.")
        return ConvSrNetVer02(**config["model"]["prior_model"])
    elif name == "Interpolator":
        logger.info("Interpolator is created.")
        return Interpolator(**config["model"]["prior_model"])
    else:
        raise NotImplementedError(f"{name} is not supported")


def make_vae_model(config: dict) -> torch.nn.Module:

    name = config["model"]["vae_model"]["name"]

    if name == "CVaeSnapshotVer02":
        logger.info("CVaeSnapshotVer02 is created.")
        return CVaeSnapshotVer02(**config["model"]["vae_model"])
    elif name == "CVaeSnapshotVer03":
        logger.info("CVaeSnapshotVer03 is created.")
        return CVaeSnapshotVer03(**config["model"]["vae_model"])
    elif name == "CVaeSnapshotVer04":
        logger.info("CVaeSnapshotVer04 is created.")
        return CVaeSnapshotVer04(**config["model"]["vae_model"])
    else:
        raise Exception(f"{name} is not supported.")


def make_vae_encoder_and_decoder(
    config: dict,
) -> typing.Tuple[torch.nn.Module, torch.nn.Module]:

    encoder, decoder = None, None

    name = config["model"]["encoder"]["name"]
    if name == "VaeEncoderVer01":
        logger.info("VaeEncoderVer01 is created.")
        encoder = VaeEncoderVer01(**config["model"]["encoder"])

    name = config["model"]["decoder"]["name"]
    if name == "VaeDecoderVer01":
        logger.info("VaeDecoderVer01 is created.")
        decoder = VaeDecoderVer01(**config["model"]["decoder"])

    return encoder, decoder