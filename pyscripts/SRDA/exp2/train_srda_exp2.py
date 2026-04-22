from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import Optional

import pytz
import torch
import torch.multiprocessing as mp
import yaml
from logging import FileHandler, StreamHandler, getLogger, INFO

# Ensure repo root is on sys.path before importing project modules.
here = Path(__file__).resolve()
root = here
for p in [here.parent, *here.parents]:
    if (p / "pyproject.toml").exists():
        root = p
        break
if str(root) not in sys.path:
    sys.path.append(str(root))

from utils.path_setup import setup_paths


os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"


def main() -> None:
    # Avoid Docker /dev/shm exhaustion when using DataLoader workers.
    try:
        mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/SRDA/exp2/srda_exp2_ldm_vqvae_ogi08.yaml",
        help="Path to SRDA Exp2 config YAML.",
    )
    args = parser.parse_args()

    # Ensure repo root and python/ are importable.
    root = setup_paths()
    root_dir = str(root)

    # Now that root/python is on sys.path, import SRDA modules.
    from src.srda.data.make_dataloaders_dict import make_dataloaders_dict
    from src.srda.utils.load_latent_model import load_latent_model
    from src.srda.utils.train_model import train_diffusion_model
    from src.srda.utils.utils import set_seeds
    from src.srda.utils.ddp import (
        cleanup_ddp,
        get_local_rank,
        get_rank,
        get_world_size,
        is_ddp,
        is_main_process,
        setup_ddp,
    )

    set_seeds(42)
    setup_ddp()
    world_size = get_world_size()
    rank = get_rank()
    local_rank = get_local_rank()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    config_path = Path(args.config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    logger = getLogger()
    if not any(isinstance(h, StreamHandler) for h in logger.handlers):
        logger.addHandler(StreamHandler(sys.stdout))
    logger.setLevel(INFO if is_main_process() else INFO + 10)

    # Resolve weight/log paths from config (they are ROOT_DIR-relative).
    weight_path = Path(root_dir + config["path"]["model"])
    logs_path = Path(root_dir + config["path"]["log"])
    weight_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    if is_main_process():
        log_file = logs_path / "train_srda_exp2.log"
        logger.addHandler(FileHandler(log_file))

    if is_main_process():
        logger.info("\n*********************************************************")
        logger.info(
            "SRDA Exp2 training start: "
            f"{datetime.datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')} ."
        )
        logger.info("*********************************************************\n")
        logger.info(f"Config path = {config_path}")
        logger.info(f"Diffusion model type = {config['diffusion_model']['model_type']}")
        logger.info(f"Model weights dir    = {weight_path}")
        logger.info(f"Logs dir             = {logs_path}")
        logger.info("\n******************** Config ********************")
        logger.info(config)
        logger.info("************************************************\n")

    if is_main_process():
        logger.info("Making SRDA dataloaders for Exp2...")
    dataloader_out = make_dataloaders_dict(
        config["datasets"],
        root_dir=root_dir,
        world_size=world_size,
        rank=rank,
        return_samplers=is_ddp(),
    )
    if is_ddp():
        dataloaders, samplers = dataloader_out
    else:
        dataloaders, samplers = dataloader_out, None

    # Latent model (required for Exp2's LDM/VQVAE setup).
    latent_model_cfg: Optional[dict] = None
    latent_cfg_path_str = config.get("path", {}).get("latent_model")
    if latent_cfg_path_str:
        latent_cfg_path = Path(root_dir + latent_cfg_path_str)
        if latent_cfg_path.exists():
            with latent_cfg_path.open("r") as f:
                latent_model_cfg = yaml.safe_load(f)

    if latent_model_cfg:
        latent_model = load_latent_model(
            config=latent_model_cfg, root_dir=root_dir, device=device
        )
    else:
        latent_model = None

    if is_main_process():
        logger.info("Starting diffusion SRDA training loop for Exp2...")
    train_diffusion_model(
        config=config,
        latent_model=latent_model,
        dataloaders=dataloaders,
        root_dir=root_dir,
        device=device,
        samplers=samplers,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    )

    if is_main_process():
        logger.info("\n*********************************************************")
        logger.info("SRDA Exp2 training finished.")
        logger.info("*********************************************************\n")
    cleanup_ddp()


if __name__ == "__main__":
    main()
