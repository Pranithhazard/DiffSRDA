import copy
import glob
import itertools
import os
import random
import re
import sys
import typing
from logging import INFO, WARNING, getLogger

import numpy as np
import torch
import torch.nn.functional as F
from src.yasuda.cfd_model.cfd.periodic_channel_domain import TorchSpectralModel2D
from src.yasuda.cfd_model.filter.low_pass_periodic_channel_domain import LowPassFilter
from src.yasuda.cfd_model.initialization.periodic_channel_jet_initializer import (
    calc_init_omega,
    calc_init_perturbation_hr_omegas,
    calc_jet_forcing,
)
from src.yasuda.cfd_model.interpolator.torch_interpolator import interpolate
from src.yasuda.dataloader import (
    make_dataloaders_vorticity_making_observation_inside_time_series_splitted,
)
from src.yasuda.dataset import (
    DatasetMakingObsInsideTimeseriesSplitted,
    DatasetMakingObsInsideTimeseriesSplittedWithMixup,
    DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling,
)
from src.yasuda.model_maker import make_model
from src.yasuda.sr_da_helper import append_zeros, inv_preprocess, preprocess

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


logger = getLogger()


def get_testdataset(
    root_dir: str,
    config: dict,
    min_start_time_index: int = 12,
    max_start_time_index: int = 999,
) -> typing.Union[
    DatasetMakingObsInsideTimeseriesSplitted,
    DatasetMakingObsInsideTimeseriesSplittedWithMixup,
    DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling,
]:
    _config = copy.deepcopy(config)
    _config["data"]["min_start_time_index"] = min_start_time_index
    _config["data"]["max_start_time_index"] = max_start_time_index

    logger.setLevel(WARNING)
    (
        dataloaders,
        _,
    ) = make_dataloaders_vorticity_making_observation_inside_time_series_splitted(
        root_dir=root_dir,
        config=_config,
        train_valid_test_kinds=["test"],
    )
    logger.setLevel(INFO)

    return dataloaders["test"].dataset


def get_testdataloader(
    root_dir: str, config: dict
) -> torch.utils.data.dataloader.DataLoader:
    logger.setLevel(WARNING)
    (
        dataloaders,
        _,
    ) = make_dataloaders_vorticity_making_observation_inside_time_series_splitted(
        root_dir=root_dir, config=config, train_valid_test_kinds=["test"]
    )
    logger.setLevel(INFO)

    return dataloaders["test"]


def make_models(config: dict, weight_path: str, cfd_config: dict):
    logger.setLevel(WARNING)

    sr_model = None
    if config is not None and weight_path is not None:
        sr_model = make_model(config).to(cfd_config["device"])
        sr_model.load_state_dict(
            torch.load(weight_path, map_location=cfd_config["device"])
        )
        _ = sr_model.eval()

    lr_model = TorchSpectralModel2D(**cfd_config)

    srda_model = TorchSpectralModel2D(**cfd_config)

    logger.setLevel(INFO)

    return sr_model, lr_model, srda_model


def initialize_models(
    t0: float,
    hr_omega0: torch.Tensor,
    lr_forcing,
    lr_model,
    srda_model,
    *,
    n_ens,
    hr_nx,
    hr_ny,
    lr_nx,
    lr_ny,
    **kwargs,
):
    assert hr_omega0.shape == (n_ens, hr_nx, hr_ny)
    omega0 = interpolate(hr_omega0, lr_nx, lr_ny, "bicubic")

    lr_ens_forcing = torch.broadcast_to(lr_forcing, (n_ens, lr_nx, lr_ny)).clone()
    assert omega0.shape == lr_ens_forcing.shape == (n_ens, lr_nx, lr_ny)

    lr_model.initialize(t0=t0, omega0=omega0, forcing=lr_ens_forcing)
    lr_model.calc_grid_data()

    if srda_model is not None:
        srda_model.initialize(t0=t0, omega0=omega0, forcing=lr_ens_forcing)
        srda_model.calc_grid_data()


def read_all_hr_omegas(
    test_dataset: typing.Union[
        DatasetMakingObsInsideTimeseriesSplitted,
        DatasetMakingObsInsideTimeseriesSplittedWithMixup,
    ]
):
    paths = test_dataset.hr_file_paths
    return _read_all_hr_omegas(paths)


def read_all_hr_omegas_with_combining(
    hr_file_paths: typing.List[str], max_ensenbles: int = 20
):
    key_func = lambda p: os.path.basename(os.path.dirname(p))
    dict_hr_paths = {
        k: sorted(g) for k, g in itertools.groupby(hr_file_paths, key_func)
    }

    all_hr_omegas = []

    for key, paths in tqdm(dict_hr_paths.items(), total=len(dict_hr_paths)):
        hr_omegas = []

        for i, p in enumerate(paths):
            all_data = []
            for j in range(max_ensenbles):
                data = np.load(p.replace("_00.npy", f"_{j:02}.npy"))
                if i > 0:
                    # skip the first time index, except for the first dataset.
                    data = data[1:]
                all_data.append(data)
            # Stack along a new dim, i.e., batch dim
            hr_omegas.append(np.stack(all_data, axis=0))

        # Concat along time dim
        hr_omegas = np.concatenate(hr_omegas, axis=1)
        all_hr_omegas.append(hr_omegas)

    # Concat along batch dim
    return torch.from_numpy(np.concatenate(all_hr_omegas, axis=0))


def read_all_hr_omegas_with_combining_for_forecast(
    hr_file_paths: typing.List[str],
    assim_period: int,
    forecast_span: int,
    max_ensenbles: int = 20,
):
    group_key = lambda p: os.path.basename(os.path.dirname(p))

    sort_key = lambda p: int(
        re.match(r"seed(\d+)_start(\d+)_end", os.path.basename(p)).groups()[1]
    )

    dict_hr_paths = {
        k: sorted(g, key=sort_key)
        for k, g in itertools.groupby(hr_file_paths, group_key)
    }

    all_hr_omegas = []

    for key, paths in tqdm(dict_hr_paths.items(), total=len(dict_hr_paths)):
        hr_omegas = []

        for i, p in enumerate(paths):
            all_data = []
            for j in range(max_ensenbles):
                data = np.load(p.replace("_00.npy", f"_{j:02}.npy"))
                if i == 0:
                    data = data[: (assim_period + forecast_span)]
                else:
                    data = data[assim_period : (assim_period + forecast_span)]
                all_data.append(data)

            # Stack along a new dim, i.e., batch dim
            hr_omegas.append(np.stack(all_data, axis=0))

        # Concat along time dim
        hr_omegas = np.concatenate(hr_omegas, axis=1)
        all_hr_omegas.append(hr_omegas)

    # Concat along batch dim
    return torch.from_numpy(np.concatenate(all_hr_omegas, axis=0))


def _read_all_hr_omegas(paths: list):
    key_func = lambda p: os.path.basename(os.path.dirname(p))
    dict_hr_paths = {k: sorted(g) for k, g in itertools.groupby(paths, key_func)}

    all_hr_omegas = []
    for key, paths in tqdm(dict_hr_paths.items(), total=len(dict_hr_paths)):
        hr_omegas = []
        for i, p in enumerate(paths):
            data = np.load(p)
            if i > 0:
                # skip the first time index, except for the first dataset.
                data = data[:, 1:]
            hr_omegas.append(data)

        # Concat along time dim
        hr_omegas = np.concatenate(hr_omegas, axis=1)
        all_hr_omegas.append(hr_omegas)

    # Concat along batch dim
    return torch.from_numpy(np.concatenate(all_hr_omegas, axis=0))


def read_all_lr_omegas(
    test_dataset: typing.Union[
        DatasetMakingObsInsideTimeseriesSplitted,
        DatasetMakingObsInsideTimeseriesSplittedWithMixup,
    ],
    lr_kind: str = "lr_omega_no-noise",
):
    paths = test_dataset.hr_file_paths
    key_func = lambda p: os.path.basename(os.path.dirname(p))
    dict_hr_paths = {k: sorted(g) for k, g in itertools.groupby(paths, key_func)}

    all_lr_omegas = []
    for key, paths in tqdm(dict_hr_paths.items(), total=len(dict_hr_paths)):
        lr_omegas = []
        for i, p in enumerate(paths):
            p = p.replace("hr_omega", lr_kind)
            assert lr_kind in p

            data = np.load(p)
            # skip the last time index, except for the last dataset.
            if i != len(paths) - 1:
                data = data[:, :-1]
                assert data.shape[1] == 4
            lr_omegas.append(data)

        # Concat along time dim
        lr_omegas = np.concatenate(lr_omegas, axis=1)
        all_lr_omegas.append(lr_omegas)

    # Concat along batch dim
    return torch.from_numpy(np.concatenate(all_lr_omegas, axis=0))


def get_observation_with_noise(
    hr_omega: torch.Tensor,
    test_dataset: typing.Union[
        DatasetMakingObsInsideTimeseriesSplitted,
        DatasetMakingObsInsideTimeseriesSplittedWithMixup,
        DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling,
    ],
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
    for _ in range(n_ens):
        _is_obses = []
        for _ in range(hr_omega.shape[1]):
            i = random.randint(0, len(test_dataset.is_obses) - 1)
            is_obs = test_dataset.is_obses[i]
            assert is_obs.shape == (hr_nx, hr_ny)
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

    if test_dataset.obs_noise_std <= 0:
        logger.info("No observation noise.")
        return hr_obsrv

    noise = np.random.normal(
        loc=0, scale=test_dataset.obs_noise_std, size=hr_obsrv.shape
    )
    logger.info(f"Observation noise std = {test_dataset.obs_noise_std}")

    return hr_obsrv + torch.from_numpy(noise)


def make_preprocessed_lr(
    lr_forecast: torch.Tensor,
    last_omega0: torch.Tensor,
    test_dataset: typing.Union[
        DatasetMakingObsInsideTimeseriesSplitted,
        DatasetMakingObsInsideTimeseriesSplittedWithMixup,
    ],
    *,
    assimilation_period,
    n_ens,
    lr_nx,
    lr_ny,
    device,
    **kwargs,
):
    lr = torch.stack(lr_forecast[-(assimilation_period + 1) :], dim=0)
    if last_omega0 is not None:
        lr[0, ...] = last_omega0
        logger.debug("last omega is added.")
    lr = lr[:: test_dataset.lr_time_interval]

    return preprocess(
        data=lr,
        biases=test_dataset.vorticity_bias,
        scales=test_dataset.vorticity_scale,
        clamp_max=test_dataset.clamp_max,
        clamp_min=test_dataset.clamp_min,
        n_ens=n_ens,
        assimilation_period=None,  # not used
        ny=lr_ny,
        nx=lr_nx,
        device=device,
    )


def make_preprocessed_obs(
    hr_obs: torch.Tensor,
    test_dataset: typing.Union[
        DatasetMakingObsInsideTimeseriesSplitted,
        DatasetMakingObsInsideTimeseriesSplittedWithMixup,
    ],
    *,
    assimilation_period,
    n_ens,
    lr_nx,
    lr_ny,
    device,
    **kwargs,
):
    obs = torch.stack(hr_obs[-(assimilation_period + 1) :], dim=0)

    obs = preprocess(
        data=obs,
        biases=test_dataset.vorticity_bias,
        scales=test_dataset.vorticity_scale,
        clamp_max=test_dataset.clamp_max,
        clamp_min=test_dataset.clamp_min,
        n_ens=n_ens,
        assimilation_period=None,  # not used
        ny=lr_ny,
        nx=lr_nx,
        device=device,
    )

    return torch.nan_to_num(obs, nan=test_dataset.missing_value)


def make_preprocessed_lr_for_forecast(
    lr_forecast: torch.Tensor,
    test_dataset: typing.Union[
        DatasetMakingObsInsideTimeseriesSplitted,
        DatasetMakingObsInsideTimeseriesSplittedWithMixup,
    ],
    *,
    assimilation_period,
    n_ens,
    lr_nx,
    lr_ny,
    device,
    **kwargs,
):
    lr = torch.stack(lr_forecast, dim=0)
    lr = lr[:: test_dataset.lr_time_interval]

    return preprocess(
        data=lr,
        biases=test_dataset.vorticity_bias,
        scales=test_dataset.vorticity_scale,
        clamp_max=test_dataset.clamp_max,
        clamp_min=test_dataset.clamp_min,
        n_ens=n_ens,
        assimilation_period=None,  # not used
        ny=lr_ny,
        nx=lr_nx,
        device=device,
    )


def make_preprocessed_obs_for_forecast(
    hr_obs: torch.Tensor,
    test_dataset: typing.Union[
        DatasetMakingObsInsideTimeseriesSplitted,
        DatasetMakingObsInsideTimeseriesSplittedWithMixup,
    ],
    *,
    assimilation_period,
    forecast_span,
    n_ens,
    lr_nx,
    lr_ny,
    device,
    **kwargs,
):
    obs = torch.stack(hr_obs[-(assimilation_period + 1) :], dim=0)
    dummy = torch.full_like(obs[0], fill_value=torch.nan)
    dummy = torch.broadcast_to(dummy, size=(forecast_span,) + dummy.shape)
    obs = torch.concat([obs, dummy], dim=0)  # stack along time

    obs = preprocess(
        data=obs,
        biases=test_dataset.vorticity_bias,
        scales=test_dataset.vorticity_scale,
        clamp_max=test_dataset.clamp_max,
        clamp_min=test_dataset.clamp_min,
        n_ens=n_ens,
        assimilation_period=None,  # not used
        ny=lr_ny,
        nx=lr_nx,
        device=device,
    )

    return torch.nan_to_num(obs, nan=test_dataset.missing_value)


def make_invprocessed_sr(
    preds: torch.Tensor,
    test_dataset: typing.Union[
        DatasetMakingObsInsideTimeseriesSplitted,
        DatasetMakingObsInsideTimeseriesSplittedWithMixup,
    ],
    *,
    assimilation_period,
    n_ens,
    hr_nx,
    hr_ny,
    **kwargs,
):
    sr = inv_preprocess(
        preds, test_dataset.vorticity_bias, test_dataset.vorticity_scale
    )

    # Delete channel dim
    sr = sr.squeeze(2)

    # ens, time, y, x -> time, ens, x, y
    sr = append_zeros(sr.permute(1, 0, 3, 2))

    assert sr.shape == (
        assimilation_period + 1,
        n_ens,
        hr_nx,
        hr_ny,
    )

    return sr


def make_invprocessed_sr_for_forecast(
    preds: torch.Tensor,
    test_dataset: typing.Union[
        DatasetMakingObsInsideTimeseriesSplitted,
        DatasetMakingObsInsideTimeseriesSplittedWithMixup,
    ],
    *,
    forecast_span,
    assimilation_period,
    n_ens,
    hr_nx,
    hr_ny,
    **kwargs,
):
    sr = inv_preprocess(
        preds, test_dataset.vorticity_bias, test_dataset.vorticity_scale
    )

    # Delete channel dim
    sr = sr.squeeze(2)

    # ens, time, y, x -> time, ens, x, y
    sr = append_zeros(sr.permute(1, 0, 3, 2))

    assert sr.shape == (
        forecast_span + assimilation_period + 1,
        n_ens,
        hr_nx,
        hr_ny,
    )

    return sr


def make_invprocessed_sr_sigma(
    preds: torch.Tensor,
    test_dataset: typing.Union[
        DatasetMakingObsInsideTimeseriesSplitted,
        DatasetMakingObsInsideTimeseriesSplittedWithMixup,
    ],
    *,
    assimilation_period,
    n_ens,
    hr_nx,
    hr_ny,
    **kwargs,
):
    sr = inv_preprocess(preds, 0.0, test_dataset.vorticity_scale)

    # Delete channel dim
    sr = sr.squeeze(2)

    # ens, time, y, x -> time, ens, x, y
    sr = append_zeros(sr.permute(1, 0, 3, 2))

    assert sr.shape == (
        assimilation_period + 1,
        n_ens,
        hr_nx,
        hr_ny,
    )

    return sr


def initialize_and_itegrate_srda_cfd_model_for_forecast(
    *,
    num_integrate_steps: int,
    last_t0: float,
    last_hr_omega0: torch.Tensor,
    lr_ens_forcing: torch.Tensor,
    cfd_config: dict,
    low_pass_filter: LowPassFilter,
):
    logger.setLevel(WARNING)
    lr_forecast = []

    srda_model = TorchSpectralModel2D(**cfd_config)

    omega0 = low_pass_filter.apply(last_hr_omega0)
    assert omega0.shape == lr_ens_forcing.shape

    srda_model.initialize(t0=last_t0, omega0=omega0, forcing=lr_ens_forcing)
    srda_model.calc_grid_data()
    lr_forecast.append(srda_model.omega.cpu().clone())

    for _ in range(num_integrate_steps):
        srda_model.time_integrate(
            dt=cfd_config["dt"], nt=cfd_config["nt"], hide_progress_bar=True
        )
        srda_model.calc_grid_data()
        lr_forecast.append(srda_model.omega.cpu().clone())

    logger.setLevel(INFO)
    return lr_forecast


def get_initial_hr_omega(
    *,
    nx: int,
    ny: int,
    num_simulations: int,
    y0: float,
    sigma: float,
    tau0: float,
    pertub_noise: float,
    u0: float,
    seed: int = 2718,
):
    hr_jet, _ = calc_jet_forcing(
        nx=nx,
        ny=ny,
        ne=num_simulations,
        y0=y0,
        sigma=sigma,
        tau0=tau0,
    )

    hr_perturb = calc_init_perturbation_hr_omegas(
        nx=nx, ny=ny, ne=num_simulations, noise_amp=pertub_noise, seed=seed
    )

    hr_omega0 = calc_init_omega(
        perturb_omega=hr_perturb,
        jet=hr_jet,
        u0=u0,
    )

    return hr_omega0


def get_uhr_and_hr_omegas(
    *,
    result_dir: str,
    uhr_nx: int,
    uhr_ny: int,
    nt: int,
    hr_nx: int,
    hr_ny: int,
):
    all_uhr_omegas = []
    for path in sorted(glob.glob(f"{result_dir}/*.npy")):
        uhr = torch.from_numpy(np.load(path)).squeeze()
        assert uhr.shape == (uhr_nx, uhr_ny)
        all_uhr_omegas.append(uhr)
    # Stack along time dim
    all_uhr_omegas = torch.stack(all_uhr_omegas)[:nt]
    assert all_uhr_omegas.shape == (nt, uhr_nx, uhr_ny)

    tmp = all_uhr_omegas[:, None, :, 1:]
    _omegas = F.avg_pool2d(tmp, kernel_size=8).squeeze()

    all_hr_omegas = torch.zeros((nt, hr_nx, hr_ny), dtype=_omegas.dtype)
    all_hr_omegas[:, :, 1:] = _omegas

    return all_uhr_omegas, all_hr_omegas
