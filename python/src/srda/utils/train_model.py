import datetime
import sys
import time
from logging import getLogger
from typing import Optional

import pandas as pd
import pytz
import src.srda.model as Model
import torch
import torch.distributed as dist
import torch.nn as nn
from src.srda.utils.ddp import is_ddp, is_main_process
from src.srda.utils.early_stopping import EarlyStopping
from src.srda.utils.utils import AverageMeter, set_seeds
from torch.nn.parallel import DistributedDataParallel as DDP

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger()


def train_diffusion_model(
    config,
    latent_model: Optional[nn.Module],
    dataloaders: dict,
    root_dir: str,
    device: str,
    samplers: Optional[dict] = None,
    world_size: int = 1,
    rank: int = 0,
    local_rank: int = 0,
    init_checkpoint_path: Optional[str] = None,
):
    del rank

    start_time = time.time()
    logger.info(
        f"Diffusion model training start: {datetime.datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')} .\n"
    )
    diffusion = Model.create_model(config, latent_model, device=device)
    diffusion.set_new_noise_schedule(config["diffusion_model"]["beta_schedule"]["train"])

    if init_checkpoint_path:
        if is_main_process():
            logger.info(f"Initializing weights from init_checkpoint_path={init_checkpoint_path}")
        state_dict = torch.load(str(init_checkpoint_path), map_location="cpu")
        diffusion.netG.load_state_dict(state_dict)

    if world_size > 1 and is_ddp():
        diffusion.netG = DDP(
            diffusion.netG,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        diffusion.optG = torch.optim.Adam(
            diffusion.netG.parameters(),
            lr=config["train_diffusion_model"]["optimizer"]["lr"],
        )

    early_stop = EarlyStopping(**config["train_diffusion_model"])
    weight_path = root_dir + config["path"]["model"]
    logs_path = root_dir + config["path"]["log"]

    epochs = config["train_diffusion_model"]["num_epochs"]
    log_every_steps = int(config.get("train_logging", {}).get("log_every_steps", 0) or 0)
    all_scores = []
    for epoch in tqdm(range(epochs), disable=not is_main_process()):
        epoch_start = time.time()
        set_seeds(epoch)

        for mode in ("train", "valid"):
            dl = dataloaders.get(mode)
            ds = getattr(dl, "dataset", None) if dl is not None else None
            if ds is not None and hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

        if samplers:
            for sampler in samplers.values():
                if sampler is not None:
                    sampler.set_epoch(epoch)

        if is_main_process():
            logger.info(f"Epoch: {epoch + 1} / {epochs}")

        losses = {}
        loss_meter = AverageMeter()
        for step, (hr_omegas, lr_omegas, lr_omega_interpotated, obs_omegas) in enumerate(
            dataloaders["train"], start=1
        ):
            if latent_model:
                inp = torch.cat((lr_omegas, obs_omegas), dim=1)
            else:
                inp = torch.cat((lr_omega_interpotated, obs_omegas), dim=1)
            diffusion.feed_data({"HR": hr_omegas, "SR": inp})
            diffusion.optimize_parameters()
            loss = diffusion.log_dict["l_pix"]
            loss_meter.update(loss, n=hr_omegas.shape[0])
            if log_every_steps > 0 and (step % log_every_steps == 0) and is_main_process():
                logger.info(
                    f"train: step {step}/{len(dataloaders['train'])} avg_loss={loss_meter.avg:.8f}"
                )

        train_avg = loss_meter.avg
        if world_size > 1 and is_ddp():
            totals = torch.tensor([loss_meter.sum, loss_meter.count], device=device, dtype=torch.float64)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            train_avg = (totals[0] / totals[1]).item()
        if is_main_process():
            logger.info(f"train: avg loss = {train_avg:.8f}")
        losses["train"] = train_avg

        if epoch % config["train_diffusion_model"]["val_freq"] == 0:
            loss_meter = AverageMeter()
            diffusion.set_new_noise_schedule(config["diffusion_model"]["beta_schedule"]["val"])
            for step, (hr_omegas, lr_omegas, lr_omega_interpotated, obs_omegas) in enumerate(
                dataloaders["valid"], start=1
            ):
                if latent_model:
                    inp = torch.cat((lr_omegas, obs_omegas), dim=1)
                else:
                    inp = torch.cat((lr_omega_interpotated, obs_omegas), dim=1)
                diffusion.feed_data({"HR": hr_omegas, "SR": inp, "True": hr_omegas})
                loss = diffusion.calc_loss_for_val()
                loss_meter.update(loss, n=hr_omegas.shape[0])
                if log_every_steps > 0 and (step % log_every_steps == 0) and is_main_process():
                    logger.info(
                        f"valid: step {step}/{len(dataloaders['valid'])} avg_loss={loss_meter.avg:.8f}"
                    )
            valid_avg = loss_meter.avg
            if world_size > 1 and is_ddp():
                totals = torch.tensor([loss_meter.sum, loss_meter.count], device=device, dtype=torch.float64)
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                valid_avg = (totals[0] / totals[1]).item()
            if is_main_process():
                logger.info(f"valid: avg loss = {valid_avg:.8f}")
            losses["valid"] = valid_avg

        all_scores.append(losses)

        should_stop = False
        if is_main_process():
            should_stop = early_stop(current_loss=losses["valid"])
        if world_size > 1 and is_ddp():
            flag = torch.tensor(int(should_stop), device=device)
            dist.broadcast(flag, src=0)
            should_stop = bool(flag.item())
        if should_stop:
            break

        if is_main_process():
            if early_stop.is_best_weight_updated:
                state_dict = (
                    diffusion.netG.module.state_dict()
                    if isinstance(diffusion.netG, DDP)
                    else diffusion.netG.state_dict()
                )
                torch.save(state_dict, weight_path + "/weight_diffusion.pth")
            if epoch % 10 == 0:
                pd.DataFrame(all_scores).to_csv(
                    logs_path + "/learn_history_diffusion.csv", index=False
                )
            logger.info(f"Elapsed time = {time.time() - epoch_start} sec")
            logger.info("*********************************************************")

    if is_main_process():
        pd.DataFrame(all_scores).to_csv(
            logs_path + "/learn_history_diffusion.csv", index=False
        )
        logger.info(f"Total elapsed time = {(time.time() - start_time) / 60.} min")
        logger.info("\nDiffusion model training finished.")


def train_latent_model(config, dataloaders: dict, root_dir: str, device: str):
    start_time = time.time()
    logger.info(
        f"Latent model training start: {datetime.datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')} .\n"
    )

    if config["latent_model"]["model_type"] != "vqvae_pixelshuffle":
        raise NotImplementedError(
            f"unknown model type: {config['latent_model']['model_type']}"
        )

    from src.srda.model.vqvae_pixelshuffle import VQVAE as latent_model_cls
    from src.srda.utils.optimize_vqvae import optimize

    latent_model = latent_model_cls(**config["latent_model"]["model"]).to(device)
    optimizer = torch.optim.Adam(
        latent_model.parameters(), lr=config["train_latent_model"]["learning_rate"]
    )
    early_stop = EarlyStopping(**config["train_latent_model"])
    weight_path = root_dir + config["path"]["model"]
    logs_path = root_dir + config["path"]["log"]

    all_scores = []
    for epoch in tqdm(range(config["train_latent_model"]["num_epochs"])):
        epoch_start = time.time()
        logger.info(f"Epoch: {epoch + 1} / {config['train_latent_model']['num_epochs']}")

        losses = {}
        for mode in ["train", "valid"]:
            losses[mode] = optimize(
                mode=mode,
                dataloader=dataloaders[mode],
                model=latent_model,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
            )
        all_scores.append(losses)

        if early_stop(current_loss=losses["valid"]):
            break

        if early_stop.is_best_weight_updated:
            torch.save(latent_model.state_dict(), weight_path + "/weight_latent.pth")

        if epoch % 10 == 0:
            pd.DataFrame(all_scores).to_csv(
                logs_path + "/learn_history_latent.csv", index=False
            )

        logger.info(f"Elapsed time = {time.time() - epoch_start} sec")
        logger.info("*********************************************************")

    pd.DataFrame(all_scores).to_csv(logs_path + "/learn_history_latent.csv", index=False)
    logger.info(f"Total elapsed time = {(time.time() - start_time) / 60.} min")
    logger.info("\nLatent model training finished.\n")
