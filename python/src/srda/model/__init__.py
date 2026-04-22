import logging

import torch
from src.srda.model.model import DDPM

logger = logging.getLogger("base")


def create_model(
    opt,
    latent_model=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    mt = opt["diffusion_model"]["model_type"]
    m = DDPM(
        opt,
        latent_model,
        device,
    )
    logger.info("Model [{:s}] is created.".format(mt.upper()))
    return m
