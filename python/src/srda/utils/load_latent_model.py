from logging import getLogger
import torch

logger = getLogger()


def load_latent_model(config, root_dir, device):
    if config["latent_model"]["model_type"] != "vqvae_pixelshuffle":
        raise NotImplementedError(
            f"unknown model type: {config['latent_model']['model_type']}"
        )

    from src.srda.model.vqvae_pixelshuffle import VQVAE as model

    latent_model = model(**config["latent_model"]["model"]).to(device)
    path = f"{root_dir}{config['path']['model']}/weight_latent.pth"
    latent_model.load_state_dict(torch.load(path, map_location=device))
    _ = latent_model.eval()
    return latent_model
