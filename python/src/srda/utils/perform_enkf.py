import sys
from logging import INFO, WARNING, StreamHandler, getLogger

logger = getLogger()
if not logger.hasHandlers():
    logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)
import gc
import glob
import os
import pathlib
import typing

import numpy as np
import torch
import yaml
from src.srda.utils.path_utils import resolve_experiment_output_root
from src.yasuda.cfd_model.cfd.periodic_channel_domain import TorchSpectralModel2D
from src.yasuda.cfd_model.enkf.sr_enkf import (
    assimilate_with_existing_data,
    calc_localization_matrix,
    hr_assimilate_with_existing_data,
)
from src.yasuda.cfd_model.filter.low_pass_periodic_channel_domain import LowPassFilter
from src.yasuda.cfd_model.initialization.periodic_channel_jet_initializer import (
    calc_init_omega,
    calc_init_perturbation_hr_omegas,
    calc_jet_forcing,
)
from src.yasuda.cfd_model.interpolator.torch_interpolator import interpolate_time_series
from src.yasuda.dataloader import split_file_paths
from src.yasuda.dataset import generate_is_obs_and_obs_matrix
from src.yasuda.utils import read_pickle, set_seeds, write_pickle
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm.notebook import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OBS_SRDA_SEED = 771155
GRID_INTERVAL = 8


def _normalize_run_suffix_for_obs_lookup(run_suffix: str) -> str:
    """
    Normalize a run suffix so we can reliably locate an existing obs `.npz` file.

    Background: some older outputs use labels like `eta1p0` while newer CLI/config
    combinations may yield `eta1`; similarly, evaluation-only runs may set a small
    `bs` that is not present in saved `.npz` filenames. For EnKF baselines we only
    need `hr_obs`, so we normalize these tokens for lookup only.
    """

    parts_out: list[str] = []
    for part in str(run_suffix).split("_"):
        if part.startswith("bs") and part[2:].isdigit():
            # Saved obs files typically do not include a batch-size token.
            continue
        if part.startswith("eta"):
            eta_raw = part[3:]
            # If `eta` has no explicit decimal (e.g. eta1), normalize to eta1p0.
            if "p" not in eta_raw and "." not in eta_raw:
                try:
                    eta_f = float(eta_raw)
                    part = "eta" + f"{eta_f:.1f}".replace(".", "p")
                except ValueError:
                    pass
        parts_out.append(part)
    return "_".join(parts_out)


def _load_hr_obs_from_seed_npz(data_dir: pathlib.Path, *, uhr_seed: int, run_suffix: str | None) -> torch.Tensor:
    """
    Load `hr_obs` from a per-seed `.npz` produced by the SRDA evaluation pipeline.

    For timing-only baseline runs (SRDA-ORG / EnKF), we prefer to reuse the exact
    synthetic observations that were already generated for the same seed/testbed.
    If the expected filename is missing (often due to minor suffix formatting
    differences), we fall back to a normalized suffix and finally to a best-effort
    glob search.
    """

    stem = f"UHR_seed_{int(uhr_seed):05}_og{GRID_INTERVAL:02}_SRDA_seed_{OBS_SRDA_SEED}"

    # 1) Direct path (if provided).
    if run_suffix:
        p = data_dir / f"{stem}_{run_suffix}.npz"
        if p.exists():
            all_data = np.load(p)
            if "hr_obs" not in all_data:
                raise KeyError(f"Missing 'hr_obs' in {p}")
            return torch.from_numpy(all_data["hr_obs"])

        # 2) Normalized suffix (eta1 -> eta1p0, drop bs tokens).
        norm = _normalize_run_suffix_for_obs_lookup(run_suffix)
        if norm != run_suffix:
            p2 = data_dir / f"{stem}_{norm}.npz"
            if p2.exists():
                all_data = np.load(p2)
                if "hr_obs" not in all_data:
                    raise KeyError(f"Missing 'hr_obs' in {p2}")
                logger.warning(
                    "Obs npz not found for run_suffix='%s'; using normalized suffix '%s' (%s).",
                    run_suffix,
                    norm,
                    str(p2),
                )
                return torch.from_numpy(all_data["hr_obs"])

    # 3) Best-effort: choose a deterministic candidate that looks like a baseline DM output.
    cands = sorted(data_dir.glob(f"{stem}_tr*_eta*.npz"))
    if not cands:
        raise FileNotFoundError(
            f"Could not find any obs npz for seed={int(uhr_seed)} in {data_dir} (stem={stem})."
        )
    chosen = cands[0]
    all_data = np.load(chosen)
    if "hr_obs" not in all_data:
        raise KeyError(f"Missing 'hr_obs' in {chosen}")
    logger.warning(
        "Obs npz not found for run_suffix='%s'; using fallback candidate (%s).",
        str(run_suffix),
        str(chosen),
    )
    return torch.from_numpy(all_data["hr_obs"])


TRAIN_VALID_TEST_RATIOS = [0.7, 0.2, 0.1]

INFLATION = 1.0
SEED = 98765
ASSIMILATION_PERIOD = 4
FORECAST_SPAN = 4
NUM_SIMULATIONS = 1

MIN_START_TIME_INDEX = -1
MAX_START_TIME_INDEX = 88
START_TIME_INDEX = 0
MAX_TIME_INDEX_FOR_INTEGRATION = 96
NUM_TIMES = MAX_START_TIME_INDEX + ASSIMILATION_PERIOD + FORECAST_SPAN

LR_NX = 32
LR_NY = 17
LR_DT = 5e-4
LR_NT = 500

HR_NX = 128
HR_NY = 65
HR_DT = LR_DT / 4.0
HR_NT = LR_NT * 4

UHR_NX = 1024
UHR_NY = 513

Y0 = np.pi / 2.0
SIGMA = 0.4
U0 = 3.0
TAU0 = 0.3
PERTUB_NOISE = 0.0025

BETA = 0.1
COEFF_LINEAR_DRAG = 1e-2
ORDER_DIFFUSION = 2
LR_COEFF_DIFFUSION = 5e-5
HR_COEFF_DIFFUSION = 1e-5

T0 = START_TIME_INDEX * LR_DT * LR_NT

INIT_TIME_INDEX = START_TIME_INDEX

ENKF_PREFERENCES = {
    4: {
        "INIT_SYS_NOISE_FACTOR": 0.0,
        "LOCALIZE_DX": 0.15,
        "N_ENS": 100,
        "OBS_PERTURB_STD": 1e-05,
        "SYS_NOISE_FACTOR": 0.0005,
    },
    6: {
        "INIT_SYS_NOISE_FACTOR": 0.2,
        "LOCALIZE_DX": 0.3,
        "N_ENS": 100,
        "OBS_PERTURB_STD": 0.03,
        "SYS_NOISE_FACTOR": 0.002,
    },
    8: {
        "INIT_SYS_NOISE_FACTOR": 0.2,
        "LOCALIZE_DX": 0.5,
        "N_ENS": 100,
        "OBS_PERTURB_STD": 0.10,
        "SYS_NOISE_FACTOR": 0.005,
    },
    10: {
        "INIT_SYS_NOISE_FACTOR": 0.2,
        "LOCALIZE_DX": 0.7,
        "N_ENS": 100,
        "OBS_PERTURB_STD": 0.20,
        "SYS_NOISE_FACTOR": 0.005,
    },
    12: {
        "INIT_SYS_NOISE_FACTOR": 0.2,
        "LOCALIZE_DX": 0.7,
        "N_ENS": 100,
        "OBS_PERTURB_STD": 0.20,
        "SYS_NOISE_FACTOR": 0.005,
    },
}

N_ENS = int(ENKF_PREFERENCES[GRID_INTERVAL]["N_ENS"])
LOCALIZE_DX = ENKF_PREFERENCES[GRID_INTERVAL]["LOCALIZE_DX"]
SYS_NOISE_FACTOR = ENKF_PREFERENCES[GRID_INTERVAL]["SYS_NOISE_FACTOR"]
INIT_SYS_NOISE_FACTOR = ENKF_PREFERENCES[GRID_INTERVAL]["INIT_SYS_NOISE_FACTOR"]
OBS_PERTURB_STD = ENKF_PREFERENCES[GRID_INTERVAL]["OBS_PERTURB_STD"]

LOCALIZE_DY = LOCALIZE_DX

HR_CFD_CONFIG = {
    "nx": HR_NX,
    "ny": HR_NY,
    "hr_nx": HR_NX,
    "hr_ny": HR_NY,
    "assimilation_period": ASSIMILATION_PERIOD,
    "coeff_linear_drag": COEFF_LINEAR_DRAG,
    "coeff_diffusion": HR_COEFF_DIFFUSION,
    "order_diffusion": ORDER_DIFFUSION,
    "beta": BETA,
    "device": DEVICE,
    "y0": Y0,
    "sigma": SIGMA,
    "tau0": TAU0,
    "t0": 0.0,
}
HR_CFD_CONFIG["ne"] = int(N_ENS)
HR_CFD_CONFIG["n_ens"] = int(N_ENS)


def get_initial_hr_omega(ne: int):
    hr_jet, _ = calc_jet_forcing(
        nx=HR_NX,
        ny=HR_NY,
        ne=ne,
        y0=Y0,
        sigma=SIGMA,
        tau0=TAU0,
    )

    hr_perturb = calc_init_perturbation_hr_omegas(
        nx=HR_NX, ny=HR_NY, ne=ne, noise_amp=PERTUB_NOISE, seed=SEED
    )

    hr_omega0 = calc_init_omega(
        perturb_omega=hr_perturb,
        jet=hr_jet,
        u0=U0,
    )

    return hr_omega0


def create_initialized_hr_model(hr_omega0: torch.Tensor, t0: float = 0.0):
    assert hr_omega0.shape == (HR_NX, HR_NY)

    omega0 = torch.broadcast_to(hr_omega0, (N_ENS, HR_NX, HR_NY))

    hr_model = TorchSpectralModel2D(**HR_CFD_CONFIG)
    _, hr_forcing = calc_jet_forcing(**HR_CFD_CONFIG)

    hr_model.initialize(t0=t0, omega0=omega0, forcing=hr_forcing)
    hr_model.calc_grid_data()

    return hr_model


def get_obs_matrices(obs_grid_interval: int):
    obs_matrices = []

    for init_x in tqdm(range(obs_grid_interval)):
        for init_y in range(obs_grid_interval):
            _, obs_mat = generate_is_obs_and_obs_matrix(
                nx=HR_NX,
                ny=HR_NY,
                init_index_x=init_x,
                init_index_y=init_y,
                interval_x=obs_grid_interval,
                interval_y=obs_grid_interval,
                dtype=torch.float64,
            )
            obs_matrices.append(obs_mat)

    return obs_matrices


def load_hr_data(
    root_dir: str,
    cfd_dir_name: str,
    train_valid_test_ratios: typing.List[str],
    kind: str,
    num_hr_omega_sets: int,
    max_ens_index: int = 20,
) -> torch.Tensor:
    cfd_dir_path = f"{root_dir}/data/ddpm/external/hr_5000_simulations_non_overlap"
    logger.info(f"CFD dir path = {cfd_dir_path}")

    data_dirs = sorted([p for p in glob.glob(f"{cfd_dir_path}/*") if os.path.isdir(p)])

    train_dirs, valid_dirs, test_dirs = split_file_paths(
        data_dirs, train_valid_test_ratios
    )

    if kind == "train":
        target_dirs = train_dirs
    elif kind == "valid":
        target_dirs = valid_dirs
    elif kind == "test":
        target_dirs = test_dirs
    else:
        raise Exception(f"{kind} is not supported.")

    logger.info(f"Kind = {kind}, Num of dirs = {len(target_dirs)}")

    all_hr_omegas = []
    for dir_path in sorted(target_dirs):
        for i in range(max_ens_index):
            hr_omegas = []
            for file_path in sorted(glob.glob(f"{dir_path}/*_hr_omega_{i:02}.npy")):
                data = np.load(file_path)

                # This is to avoid overlapping at the start/end point
                if len(hr_omegas) > 0:
                    data = data[1:]
                hr_omegas.append(data)

            # Concat along time axis
            all_hr_omegas.append(np.concatenate(hr_omegas, axis=0))

            if len(all_hr_omegas) == num_hr_omega_sets:
                # Concat along batch axis
                ret = np.stack(all_hr_omegas, axis=0)
                return torch.from_numpy(ret).to(torch.float64)

    ret = np.stack(all_hr_omegas, axis=0)
    return torch.from_numpy(ret).to(torch.float64)


def get_obs_matrix(obs: torch.Tensor) -> torch.Tensor:
    assert obs.shape == (HR_NX, HR_NY)
    is_obs = torch.where(torch.isnan(obs), torch.zeros_like(obs), torch.ones_like(obs))

    obs_indices = is_obs.reshape(-1)
    obs_indices = torch.where(obs_indices == 1.0)[0]

    num_obs = len(obs_indices)

    obs_matrix = torch.zeros(num_obs, HR_NX * HR_NY, dtype=torch.float64, device=DEVICE)

    for i, j in enumerate(obs_indices):
        obs_matrix[i, j] = 1.0

    p = 100 * torch.sum(obs_matrix).item() / (HR_NX * HR_NY)
    logger.debug(f"observatio prob = {p} [%]")

    return obs_matrix


def get_sys_noise_generator(root_dir, num_hr_omega_sets: int = 250, eps: float = 1e-12):
    hr_omegas = load_hr_data(
        root_dir=root_dir,
        cfd_dir_name=None,
        train_valid_test_ratios=TRAIN_VALID_TEST_RATIOS,
        kind="train",
        num_hr_omega_sets=num_hr_omega_sets,
    )
    # dims = batch, time, x, and y
    logger.info(f"hr_omega shape = {hr_omegas.shape}")

    lr_omegas = interpolate_time_series(hr_omegas, LR_NX, LR_NY, "bicubic")
    lr_omegas = lr_omegas - torch.mean(lr_omegas, dim=0, keepdim=True)

    lr_omegas = lr_omegas.reshape(lr_omegas.shape[:2] + (-1,))
    # dims = batch, time, and space

    del hr_omegas
    gc.collect()

    # Inner product over batch dim
    all_covs = torch.mean(lr_omegas[..., None, :] * lr_omegas[..., None], dim=0)

    # Assure conv is symmetric.
    all_covs = (all_covs + all_covs.permute(0, 2, 1)) / 2.0

    # Assure positive definiteness
    all_covs = all_covs + torch.diag(
        torch.full(size=(all_covs.shape[-1],), fill_value=eps)
    )

    loc = torch.zeros(all_covs.shape[-1], dtype=torch.float64)
    return [MultivariateNormal(loc, cov) for cov in all_covs]


def get_cov_for_sys_noise_generator(
    root_dir: str, num_hr_omega_sets: int = 50, eps: float = 1e-10
):
    hr_omegas = load_hr_data(
        root_dir=root_dir,
        cfd_dir_name=None,
        train_valid_test_ratios=TRAIN_VALID_TEST_RATIOS,
        kind="train",
        num_hr_omega_sets=num_hr_omega_sets,
    )
    # dims = batch, time, x, and y

    hr_omegas = hr_omegas.reshape(hr_omegas.shape[:2] + (-1,))
    hr_omegas = hr_omegas[:, ::ASSIMILATION_PERIOD]

    # Inner product over batch dim
    all_covs = torch.mean(hr_omegas[..., None, :] * hr_omegas[..., None], dim=0)

    # Assure conv is symmetric.
    all_covs = (all_covs + all_covs.permute(0, 2, 1)) / 2.0

    # Assure positive definiteness
    all_covs = all_covs + torch.diag(
        torch.full(size=(all_covs.shape[-1],), fill_value=eps)
    )

    return all_covs


def perform_enkf_hr(
    i_seed_start,
    i_seed_end,
    root_dir,
    experiment_name,
    test_name,
    device,
    processed_base_dir: typing.Optional[str] = None,
    run_suffix: typing.Optional[str] = None,
    *,
    timing: bool = False,
    timing_out_dir: typing.Optional[str] = None,
    timing_warmup_cycles: int = 0,
    timing_skip_save_outputs: bool = False,
    timing_run_suffix: typing.Optional[str] = None,
):
    CONFIG_PATH = f"{root_dir}/python/configs/srda/{experiment_name}/{test_name}.yaml"

    with open(CONFIG_PATH, "r") as file:
        CONFIG = yaml.safe_load(file)
    CONFIG_INFO = {
        "config": CONFIG,
        "weight_path": f"{root_dir}{CONFIG['path']['model']}/weight_diffusion.pth",
    }

    HR_CFD_CONFIG = {
        "nx": HR_NX,
        "ny": HR_NY,
        "hr_nx": HR_NX,
        "hr_ny": HR_NY,
        "assimilation_period": ASSIMILATION_PERIOD,
        "coeff_linear_drag": COEFF_LINEAR_DRAG,
        "coeff_diffusion": HR_COEFF_DIFFUSION,
        "order_diffusion": ORDER_DIFFUSION,
        "beta": BETA,
        "device": device,
        "y0": Y0,
        "sigma": SIGMA,
        "tau0": TAU0,
        "t0": 0.0,
    }
    HR_CFD_CONFIG["ne"] = int(N_ENS)
    HR_CFD_CONFIG["n_ens"] = int(N_ENS)
    cov_file_path = f"{root_dir}/data/srda/processed/enkf_hr/sys_noise_covs.pickle"

    if os.path.exists(cov_file_path):
        all_covs = read_pickle(cov_file_path)
    else:
        os.makedirs(os.path.dirname(cov_file_path), exist_ok=True)
        all_covs = get_cov_for_sys_noise_generator(root_dir=root_dir)
        write_pickle(all_covs, cov_file_path)
    torch_rand_generator = torch.Generator().manual_seed(SEED)

    loc = torch.zeros(all_covs.shape[-1], dtype=torch.float64)
    sys_noise_generators = [MultivariateNormal(loc, cov) for cov in all_covs]
    sys_noise_generators = sys_noise_generators[
        INIT_TIME_INDEX // ASSIMILATION_PERIOD:
    ]

    localization_matrix = calc_localization_matrix(
        nx=HR_NX, ny=HR_NY, d_x=LOCALIZE_DX, d_y=LOCALIZE_DY
    ).cpu()

    LR_CFD_CONFIG = {
        "nx": LR_NX,
        "ny": LR_NY,
        "lr_nx": LR_NX,
        "lr_ny": LR_NY,
        "hr_nx": HR_NX,
        "hr_ny": HR_NY,
        "assimilation_period": ASSIMILATION_PERIOD,
        "coeff_linear_drag": COEFF_LINEAR_DRAG,
        "coeff_diffusion": LR_COEFF_DIFFUSION,
        "order_diffusion": ORDER_DIFFUSION,
        "beta": BETA,
        "device": device,
        "y0": Y0,
        "sigma": SIGMA,
        "tau0": TAU0,
        "t0": 0.0,
    }
    LR_CFD_CONFIG["ne"] = int(N_ENS)
    LR_CFD_CONFIG["n_ens"] = int(N_ENS)

    logger.setLevel(WARNING)
    lr_model = TorchSpectralModel2D(**LR_CFD_CONFIG)
    _, lr_forcing = calc_jet_forcing(**LR_CFD_CONFIG)
    logger.setLevel(INFO)
    assert lr_forcing.shape == (100, LR_NX, LR_NY)

    exp_root = resolve_experiment_output_root(
        root_dir=root_dir,
        experiment_name=experiment_name,
        processed_base_dir=processed_base_dir,
    )
    suffix_str = f"_{run_suffix}" if run_suffix else ""

    timing_enabled = bool(timing)
    timing_recorder = None
    if timing_enabled:
        from src.srda.utils.timing import (
            CudaWallTimer,
            CycleTimingRow,
            RunTimingSummary,
            TimingRecorder,
        )

        base_dir = (
            pathlib.Path(timing_out_dir)
            if timing_out_dir
            else (exp_root / "timing" / test_name)
        )
        run_id_suffix = str(timing_run_suffix or "enkf_hr")
        run_id = f"{run_id_suffix}_seeds{i_seed_start:05d}-{i_seed_end:05d}"
        timing_run_dir = base_dir / run_id
        meta = RunTimingSummary(
            experiment_name=str(experiment_name),
            test_name=str(test_name),
            seeds_start=int(i_seed_start),
            seeds_end=int(i_seed_end),
            run_suffix=str(run_id_suffix),
            respacing=None,
            eta=None,
            batch_size=None,
            obs_guidance_mode=None,
            obs_guidance_space=None,
            extra={
                "method": "enkf_hr",
                "timing_skip_save_outputs": bool(timing_skip_save_outputs),
            },
        )
        timing_recorder = TimingRecorder(
            out_dir=timing_run_dir,
            run_meta=meta,
            warmup_cycles=int(timing_warmup_cycles),
        )

    for i_seed_uhr in range(i_seed_start, i_seed_end + 1):
        seed_cycle_rows = []
        if timing_enabled:
            seed_timer = CudaWallTimer(device=device, enabled=True)
            seed_timer.start()

        data_dir = exp_root / "data" / test_name / f"{i_seed_uhr:05}"
        csv_dir = exp_root / "csv" / test_name / f"{i_seed_uhr:05}"
        fig_dir = exp_root / "fig" / test_name / f"{i_seed_uhr:05}"
        for _dir in (data_dir, csv_dir, fig_dir):
            os.makedirs(_dir, exist_ok=True)

        output_hr_enkf_file_path = (
            data_dir
            / f"UHR_seed_{i_seed_uhr:05}_og{GRID_INTERVAL:02}_SRDA_seed_{OBS_SRDA_SEED}_enkf_hr_mean{suffix_str}.npy"
        )
        output_hr_enkf_files_path = (
            data_dir
            / f"UHR_seed_{i_seed_uhr:05}_og{GRID_INTERVAL:02}_SRDA_seed_{OBS_SRDA_SEED}_enkf_hr_ens{suffix_str}.npy"
        )

        hr_obs = _load_hr_obs_from_seed_npz(
            data_dir,
            uhr_seed=int(i_seed_uhr),
            run_suffix=str(run_suffix) if run_suffix else None,
        )

        set_seeds(SEED, use_deterministic=not timing_enabled)
        init_hr_omega = get_initial_hr_omega(ne=1).squeeze()

        hr_model = create_initialized_hr_model(init_hr_omega)
        hr_enkfs = []
        cnt_assim = 0

        cycle_timer = None
        cycle_start_i = None
        for i_cycle in tqdm(range(NUM_TIMES), disable=timing_enabled):
            # Data assimilation
            if i_cycle > 0 and i_cycle % ASSIMILATION_PERIOD == 0:
                if timing_enabled and timing_recorder is not None and cycle_timer is not None:
                    cycle_total_sec = cycle_timer.stop()
                    row = CycleTimingRow(
                        seed=int(i_seed_uhr),
                        i_cycle=int(cycle_start_i),
                        cycle_index=int(cycle_start_i // ASSIMILATION_PERIOD),
                        sampling_steps=0,
                        cycle_total_sec=float(cycle_total_sec),
                        sampling_sec=0.0,
                        sampling_ms_per_step=float("nan"),
                        lr_forecast_sec=float("nan"),
                    )
                    seed_cycle_rows.append(row)
                    timing_recorder.add_cycle_row(row)

                if timing_enabled and timing_recorder is not None:
                    cycle_timer = CudaWallTimer(device=device, enabled=True)
                    cycle_timer.start()
                    cycle_start_i = int(i_cycle)

                obs = hr_obs[i_cycle].to(torch.float64)
                obs_matrix = get_obs_matrix(obs)

                # This is to avoid nan when observation operator acts.
                obs = torch.nan_to_num(obs, nan=1e10)

                # This method returns forecast conv
                hr_assimilate_with_existing_data(
                    hr_omega=obs.to(DEVICE),
                    hr_ens_model=hr_model,
                    obs_matrix=obs_matrix,
                    obs_noise_std=OBS_PERTURB_STD,
                    inflation=INFLATION,
                    rand_generator=torch_rand_generator,
                    localization_matrix=localization_matrix,
                )

            # Store data "after" assimilation
            hr_enkfs.append(hr_model.omega.cpu().clone())

            # Add additive system noise
            if i_cycle == 0 or (
                INFLATION == 1.0 and i_cycle % ASSIMILATION_PERIOD == 0
            ):
                noise = sys_noise_generators[cnt_assim].sample([N_ENS])
                noise = noise.reshape(N_ENS, HR_NX, HR_NY)
                noise = noise - torch.mean(noise, dim=0, keepdims=True)
                assert noise.shape == hr_model.omega.shape

                factor = INIT_SYS_NOISE_FACTOR if i_cycle == 0 else SYS_NOISE_FACTOR
                omega = hr_model.omega + factor * noise.to(DEVICE)

                hr_model.initialize(t0=hr_model.t, omega0=omega)
                hr_model.calc_grid_data()
                cnt_assim += 1

            hr_model.time_integrate(dt=HR_DT, nt=HR_NT, hide_progress_bar=True)
            hr_model.calc_grid_data()

        if timing_enabled and timing_recorder is not None and cycle_timer is not None:
            cycle_total_sec = cycle_timer.stop()
            row = CycleTimingRow(
                seed=int(i_seed_uhr),
                i_cycle=int(cycle_start_i),
                cycle_index=int(cycle_start_i // ASSIMILATION_PERIOD),
                sampling_steps=0,
                cycle_total_sec=float(cycle_total_sec),
                sampling_sec=0.0,
                sampling_ms_per_step=float("nan"),
                lr_forecast_sec=float("nan"),
            )
            seed_cycle_rows.append(row)
            timing_recorder.add_cycle_row(row)

        # Stack along time dim
        hr_enkfs = torch.stack(hr_enkfs, dim=1).to(torch.float32)

        # Add an ensemble mean
        hr_mean_enkf = torch.mean(hr_enkfs, dim=0).numpy()
        assert hr_mean_enkf.shape == (NUM_TIMES, HR_NX, HR_NY)

        if not timing_skip_save_outputs:
            np.save(output_hr_enkf_file_path, hr_mean_enkf)
            np.save(output_hr_enkf_files_path, hr_enkfs.numpy())

        if timing_enabled and timing_recorder is not None:
            seed_total_sec = seed_timer.stop()
            timing_recorder.add_seed_summary(
                seed=int(i_seed_uhr),
                total_sec=float(seed_total_sec),
                rows_for_seed=seed_cycle_rows,
            )

    if timing_enabled and timing_recorder is not None:
        timing_recorder.write()


low_pass_filter = LowPassFilter(
    nx_lr=LR_NX, ny_lr=LR_NY, nx_hr=HR_NX, ny_hr=HR_NY, device=DEVICE
)


def initialize_lr_model(
    *,
    t0: float,
    hr_omega0: torch.Tensor,
    lr_forcing: torch.Tensor,
    lr_model: TorchSpectralModel2D,
    n_ens: int,
    hr_nx: int,
    hr_ny: int,
    lr_nx: int,
    lr_ny: int,
    **kwargs,
):
    assert hr_omega0.shape == (hr_nx, hr_ny)
    omega0 = low_pass_filter.apply(hr_omega0[None, ...])
    omega0 = torch.broadcast_to(omega0, (n_ens, lr_nx, lr_ny))

    lr_model.initialize(t0=t0, omega0=omega0, forcing=lr_forcing)
    lr_model.calc_grid_data()


def perform_enkf_bicubic(
    i_seed_start,
    i_seed_end,
    root_dir,
    experiment_name,
    test_name,
    device,
    processed_base_dir: typing.Optional[str] = None,
    run_suffix: typing.Optional[str] = None,
    *,
    timing: bool = False,
    timing_out_dir: typing.Optional[str] = None,
    timing_warmup_cycles: int = 0,
    timing_skip_save_outputs: bool = False,
    timing_run_suffix: typing.Optional[str] = None,
):
    sys_noise_generators = get_sys_noise_generator(root_dir=root_dir)
    _ = gc.collect()
    torch_rand_generator = torch.Generator().manual_seed(SEED)

    localization_matrix = calc_localization_matrix(
        nx=HR_NX, ny=HR_NY, d_x=LOCALIZE_DX, d_y=LOCALIZE_DY
    ).to(device)

    HR_CFD_CONFIG = {
        "nx": HR_NX,
        "ny": HR_NY,
        "hr_nx": HR_NX,
        "hr_ny": HR_NY,
        "assimilation_period": ASSIMILATION_PERIOD,
        "coeff_linear_drag": COEFF_LINEAR_DRAG,
        "coeff_diffusion": HR_COEFF_DIFFUSION,
        "order_diffusion": ORDER_DIFFUSION,
        "beta": BETA,
        "device": DEVICE,
        "y0": Y0,
        "sigma": SIGMA,
        "tau0": TAU0,
        "t0": 0.0,
    }
    HR_CFD_CONFIG["ne"] = int(N_ENS)
    HR_CFD_CONFIG["n_ens"] = int(N_ENS)

    LR_CFD_CONFIG = {
        "nx": LR_NX,
        "ny": LR_NY,
        "lr_nx": LR_NX,
        "lr_ny": LR_NY,
        "hr_nx": HR_NX,
        "hr_ny": HR_NY,
        "assimilation_period": ASSIMILATION_PERIOD,
        "coeff_linear_drag": COEFF_LINEAR_DRAG,
        "coeff_diffusion": LR_COEFF_DIFFUSION,
        "order_diffusion": ORDER_DIFFUSION,
        "beta": BETA,
        "device": device,
        "y0": Y0,
        "sigma": SIGMA,
        "tau0": TAU0,
        "t0": 0.0,
    }
    LR_CFD_CONFIG["ne"] = int(N_ENS)
    LR_CFD_CONFIG["n_ens"] = int(N_ENS)
    logger.setLevel(WARNING)
    lr_model = TorchSpectralModel2D(**LR_CFD_CONFIG)
    _, lr_forcing = calc_jet_forcing(**LR_CFD_CONFIG)
    logger.setLevel(INFO)
    assert lr_forcing.shape == (100, LR_NX, LR_NY)

    exp_root = resolve_experiment_output_root(
        root_dir=root_dir,
        experiment_name=experiment_name,
        processed_base_dir=processed_base_dir,
    )
    suffix_str = f"_{run_suffix}" if run_suffix else ""

    timing_enabled = bool(timing)
    timing_recorder = None
    if timing_enabled:
        from src.srda.utils.timing import (
            CudaWallTimer,
            CycleTimingRow,
            RunTimingSummary,
            TimingRecorder,
        )

        base_dir = (
            pathlib.Path(timing_out_dir)
            if timing_out_dir
            else (exp_root / "timing" / test_name)
        )
        run_id_suffix = str(timing_run_suffix or "enkf_bicubic")
        run_id = f"{run_id_suffix}_seeds{i_seed_start:05d}-{i_seed_end:05d}"
        timing_run_dir = base_dir / run_id
        meta = RunTimingSummary(
            experiment_name=str(experiment_name),
            test_name=str(test_name),
            seeds_start=int(i_seed_start),
            seeds_end=int(i_seed_end),
            run_suffix=str(run_id_suffix),
            respacing=None,
            eta=None,
            batch_size=None,
            obs_guidance_mode=None,
            obs_guidance_space=None,
            extra={
                "method": "enkf_bicubic",
                "timing_skip_save_outputs": bool(timing_skip_save_outputs),
            },
        )
        timing_recorder = TimingRecorder(
            out_dir=timing_run_dir,
            run_meta=meta,
            warmup_cycles=int(timing_warmup_cycles),
        )

    for i_seed_uhr in range(i_seed_start, i_seed_end + 1):
        seed_cycle_rows = []
        if timing_enabled:
            seed_timer = CudaWallTimer(device=device, enabled=True)
            seed_timer.start()

        data_dir = exp_root / "data" / test_name / f"{i_seed_uhr:05}"
        csv_dir = exp_root / "csv" / test_name / f"{i_seed_uhr:05}"
        fig_dir = exp_root / "fig" / test_name / f"{i_seed_uhr:05}"
        for _dir in (data_dir, csv_dir, fig_dir):
            os.makedirs(_dir, exist_ok=True)
        output_lr_file_path = (
            data_dir
            / f"UHR_seed_{i_seed_uhr:05}_og{GRID_INTERVAL:02}_SRDA_seed_{OBS_SRDA_SEED}_ens_bicubic_lr{suffix_str}.npy"
        )
        output_hr_file_path = (
            data_dir
            / f"UHR_seed_{i_seed_uhr:05}_og{GRID_INTERVAL:02}_SRDA_seed_{OBS_SRDA_SEED}_ens_bicubic_mean_hr{suffix_str}.pickle"
        )

        hr_obs = _load_hr_obs_from_seed_npz(
            data_dir,
            uhr_seed=int(i_seed_uhr),
            run_suffix=str(run_suffix) if run_suffix else None,
        )

        set_seeds(SEED, use_deterministic=not timing_enabled)
        init_hr_omega = get_initial_hr_omega(ne=1).squeeze()

        initialize_lr_model(
            hr_omega0=init_hr_omega,
            lr_forcing=lr_forcing,
            lr_model=lr_model,
            **LR_CFD_CONFIG,
        )
        lr_enkfs = []
        dict_hr_analysis = {}

        cycle_timer = None
        cycle_start_i = None
        for i_cycle in tqdm(range(MAX_TIME_INDEX_FOR_INTEGRATION), disable=timing_enabled):
            # Data assimilation
            if i_cycle > 0 and i_cycle % ASSIMILATION_PERIOD == 0:
                if timing_enabled and timing_recorder is not None and cycle_timer is not None:
                    cycle_total_sec = cycle_timer.stop()
                    row = CycleTimingRow(
                        seed=int(i_seed_uhr),
                        i_cycle=int(cycle_start_i),
                        cycle_index=int(cycle_start_i // ASSIMILATION_PERIOD),
                        sampling_steps=0,
                        cycle_total_sec=float(cycle_total_sec),
                        sampling_sec=0.0,
                        sampling_ms_per_step=float("nan"),
                        lr_forecast_sec=float("nan"),
                    )
                    seed_cycle_rows.append(row)
                    timing_recorder.add_cycle_row(row)

                if timing_enabled and timing_recorder is not None:
                    cycle_timer = CudaWallTimer(device=device, enabled=True)
                    cycle_timer.start()
                    cycle_start_i = int(i_cycle)

                obs = hr_obs[i_cycle].to(torch.float64)
                obs_matrix = get_obs_matrix(obs)

                # This is to avoid nan when observation operator acts.
                obs = torch.nan_to_num(obs, nan=1e10)

                all_hr_analysis, _ = assimilate_with_existing_data(
                    hr_omega=obs.to(DEVICE),
                    lr_ens_model=lr_model,
                    obs_matrix=obs_matrix,
                    obs_noise_std=OBS_PERTURB_STD,
                    inflation=INFLATION,
                    rand_generator=torch_rand_generator,
                    localization_matrix=localization_matrix,
                    return_hr_analysis=True,
                )

                # Mean over batch (ensemble dim)
                hr_analysis = torch.mean(all_hr_analysis, axis=0)
                assert hr_analysis.shape == (HR_NX, HR_NY)
                dict_hr_analysis[i_cycle] = hr_analysis.cpu().to(torch.float32).numpy()

            lr_enkfs.append(lr_model.omega.cpu().clone())

            # Add additive system noise
            if i_cycle == 0 or (INFLATION == 1.0 and i_cycle % ASSIMILATION_PERIOD == 0):
                noise = sys_noise_generators[START_TIME_INDEX + i_cycle].sample([N_ENS])
                noise = noise.reshape(N_ENS, LR_NX, LR_NY)
                noise = noise - torch.mean(noise, dim=0, keepdims=True)

                factor = INIT_SYS_NOISE_FACTOR if i_cycle == 0 else SYS_NOISE_FACTOR
                omega = lr_model.omega + factor * noise.to(DEVICE)

                lr_model.initialize(t0=lr_model.t, omega0=omega)
                lr_model.calc_grid_data()

            lr_model.time_integrate(dt=LR_DT, nt=LR_NT, hide_progress_bar=True)
            lr_model.calc_grid_data()

        if timing_enabled and timing_recorder is not None and cycle_timer is not None:
            cycle_total_sec = cycle_timer.stop()
            row = CycleTimingRow(
                seed=int(i_seed_uhr),
                i_cycle=int(cycle_start_i),
                cycle_index=int(cycle_start_i // ASSIMILATION_PERIOD),
                sampling_steps=0,
                cycle_total_sec=float(cycle_total_sec),
                sampling_sec=0.0,
                sampling_ms_per_step=float("nan"),
                lr_forecast_sec=float("nan"),
            )
            seed_cycle_rows.append(row)
            timing_recorder.add_cycle_row(row)

        if not timing_skip_save_outputs:
            write_pickle(dict_hr_analysis, output_hr_file_path)

        # Stack along time dim
        lr_enkfs = torch.stack(lr_enkfs, dim=1).to(torch.float32).numpy()
        if not timing_skip_save_outputs:
            np.save(output_lr_file_path, lr_enkfs)

        if timing_enabled and timing_recorder is not None:
            seed_total_sec = seed_timer.stop()
            timing_recorder.add_seed_summary(
                seed=int(i_seed_uhr),
                total_sec=float(seed_total_sec),
                rows_for_seed=seed_cycle_rows,
            )

    if timing_enabled and timing_recorder is not None:
        timing_recorder.write()
