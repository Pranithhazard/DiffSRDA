import random
import sys
from logging import DEBUG, INFO, WARNING, StreamHandler, getLogger
from src.srda.data.dataset import SrdaByDdpmDataset
import numpy as np
import torch
import torch.nn.functional as F

logger = getLogger()
if not logger.hasHandlers():
    logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)


def make_preprocessed_lr_for_forecast(
    lr_forecast: torch.Tensor, dataset: SrdaByDdpmDataset, use_lr: bool
):
    lr = torch.stack(lr_forecast, dim=0)  # stack along time
    lr = lr.squeeze()
    logger.debug(f"LR shape before sampling = {lr.shape}")
    lr = lr[:: dataset.obs_time_interval]
    logger.debug(f"LR shape after sampling = {lr.shape}")

    lr = dataset.normalize(lr)
    lr = dataset.drop_y_boundary(lr)
    if not use_lr:
        lr = F.interpolate(
            lr[None, ...], scale_factor=dataset.scale_factor, mode="bicubic"
        ).squeeze(0)

    return lr


def make_preprocessed_obs_for_forecast_with_raw(
    hr_obs: torch.Tensor,
    dataset: SrdaByDdpmDataset,
    assimilation_period: int,
    use_lr: bool,
):
    logger.debug(f"Input assimilation period = {assimilation_period}")

    # Past points are extracted, whose number is `assimilation_period + 1`
    obs = hr_obs[-(assimilation_period + 1) :]
    logger.debug(f"Obs length after extraction = {len(obs)}")

    # Subsampling obs
    obs = obs[:: dataset.obs_time_interval]
    logger.debug(f"Obs length after subsampling = {len(obs)}")

    # obs is stacked along time axis.
    obs = torch.stack(obs, dim=0)
    logger.debug(f"Obs shape after stacking = {obs.shape}")

    obs = dataset.normalize(obs.to(dataset.dtype))
    obs = dataset.drop_y_boundary(obs)
    obs_raw = obs.clone()
    if use_lr:
        obs = dataset.process_obs(obs)
        obs = dataset.max_pool(obs)
    else:
        obs = dataset.process_obs(obs)
    return obs, obs_raw


def make_preprocessed_obs_for_forecast(
    hr_obs: torch.Tensor,
    dataset: SrdaByDdpmDataset,
    assimilation_period: int,
    use_lr: bool,
):
    obs, _ = make_preprocessed_obs_for_forecast_with_raw(
        hr_obs=hr_obs,
        dataset=dataset,
        assimilation_period=assimilation_period,
        use_lr=use_lr,
    )
    return obs


def get_observation_with_noise(
    hr_omega: torch.Tensor,
    dataset: SrdaByDdpmDataset,
    *,
    n_ens,
    hr_nx,
    hr_ny,
    lr_nx,
    lr_ny,
    **kwargs,
) -> torch.Tensor:
    assert hr_omega.ndim == 4  # ens, time, x, y dims
    assert hr_omega.shape[0] == n_ens
    assert hr_omega.shape[2] == hr_nx
    assert hr_omega.shape[3] == hr_ny

    is_obses = []
    for _ in range(n_ens):  # batch
        _is_obses = []
        for _ in range(hr_omega.shape[1]):  # time
            i = random.randint(0, len(dataset.is_obses) - 1)
            is_obs = torch.zeros(hr_nx, hr_ny)
            is_obs[:, :-1] = dataset.is_obses[i]
            assert is_obs.shape == (hr_nx, hr_ny), f"shape = {is_obs.shape}"
            _is_obses.append(is_obs)

        is_obses.append(torch.stack(_is_obses, dim=0))

    is_obses = torch.stack(is_obses, dim=0)
    assert is_obses.shape == hr_omega.shape

    hr_obsrv = torch.full_like(hr_omega, np.nan)
    hr_obsrv = torch.where(
        is_obses > 0,
        hr_omega,
        hr_obsrv,
    )

    if dataset.obs_noise_std <= 0:
        logger.info("No observation noise.")
        return hr_obsrv

    noise = np.random.normal(loc=0, scale=dataset.obs_noise_std, size=hr_obsrv.shape)
    logger.info(f"Observation noise std = {dataset.obs_noise_std}")

    return hr_obsrv + torch.from_numpy(noise)


def make_invprocessed_sr_for_forecast(preds: torch.Tensor, dataset: SrdaByDdpmDataset):
    out = dataset.inv_normalize(preds)
    shape = (out.shape[0], out.shape[1], out.shape[2], out.shape[3] + 1)
    sr = torch.zeros(shape, dtype=out.dtype)
    sr[:, :, :, :-1] = out
    sr[:, :, :, -1] = out[:, :, :, 0]
    return sr
