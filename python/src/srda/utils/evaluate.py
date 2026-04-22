import sys
from logging import DEBUG, INFO, WARNING, StreamHandler, getLogger

logger = getLogger()
if not logger.hasHandlers():
    logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

import gc
import glob
import os
import pathlib
import random
import time
from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
import src.srda.model as Model
import torch
import torch.nn.functional as F
import yaml
from src.srda.data.dataloader import make_dataloaders
from src.srda.data.dataset import SrdaByDdpmDataset
from src.srda.data.make_dataloaders_dict import make_dataloaders_dict
from src.srda.utils.load_latent_model import load_latent_model
from src.srda.utils.sr_da_dm_helper import (
    get_observation_with_noise,
    make_invprocessed_sr_for_forecast,
    make_preprocessed_lr_for_forecast,
    make_preprocessed_obs_for_forecast_with_raw,
)
from src.srda.utils.timing import (
    CudaWallTimer,
    CycleTimingRow,
    RunTimingSummary,
    TimingRecorder,
)
from src.srda.utils.sensor_scenarios import (
    SensorScenarioConfig,
    generate_sensor_scenario_hr_observations,
)
from src.yasuda.cfd_model.cfd.periodic_channel_domain import TorchSpectralModel2D
from src.yasuda.cfd_model.filter.low_pass_periodic_channel_domain import LowPassFilter
from src.yasuda.cfd_model.initialization.periodic_channel_jet_initializer import (
    calc_jet_forcing,
)
from src.yasuda.sr_da_helper_2 import (
    get_initial_hr_omega,
    get_uhr_and_hr_omegas,
    initialize_and_itegrate_srda_cfd_model_for_forecast,
)
from src.yasuda.utils import set_seeds
from tqdm.notebook import tqdm
from src.srda.utils.path_utils import resolve_experiment_output_root

OBS_SRDA_SEED = 771155
GRID_INTERVAL = 8

ASSIMILATION_PERIOD = 4
FORECAST_SPAN = 4
NUM_SIMULATIONS = 1

MIN_START_TIME_INDEX = -1
MAX_START_TIME_INDEX = 88
START_TIME_INDEX = 0
NUM_TIMES = MAX_START_TIME_INDEX + ASSIMILATION_PERIOD + FORECAST_SPAN

LR_NX = 32
LR_NY = 17
LR_DT = 5e-4
LR_NT = 500

HR_NX = 128
HR_NY = 65

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

T0 = START_TIME_INDEX * LR_DT * LR_NT


DEFAULT_GUIDANCE_SIGMA = 0.05


def _infer_sampling_steps(timestep_respacing: Optional[Union[int, list, tuple]]) -> int:
    if timestep_respacing is None:
        return 0
    if isinstance(timestep_respacing, (list, tuple)):
        total = 0
        for v in timestep_respacing:
            try:
                total += int(v)
            except (TypeError, ValueError):
                continue
        return int(total)
    try:
        return int(timestep_respacing)
    except (TypeError, ValueError):
        return 0


def _prepare_obs_guidance(
    mode: str,
    obs_raw: torch.Tensor,
    sr_channels: int,
    dtype: torch.dtype,
    gamma: float,
    sigma: Optional[float],
    apply_every: int,
    apply_during_sampling: bool,
    blur_sigma_px: float,
    recompute_eps: bool,
    tighten_final_steps: int,
    blur_sigma_px_final: Optional[float],
    blur_schedule_power: float,
) -> Optional[dict]:
    if mode == "off":
        return None

    obs_curr = obs_raw[-1]
    mask2d = torch.isfinite(obs_curr)
    if not torch.any(mask2d):
        return None

    target2d = torch.where(mask2d, obs_curr, torch.zeros_like(obs_curr))
    mask = torch.zeros(
        (1, sr_channels, obs_curr.shape[-2], obs_curr.shape[-1]), dtype=torch.bool
    )
    mask[:, 0] = mask2d
    target = torch.zeros(
        (1, sr_channels, obs_curr.shape[-2], obs_curr.shape[-1]), dtype=dtype
    )
    target[:, 0] = target2d

    return {
        "mode": mode,
        "mask": mask,
        "target": target,
        "gamma": float(gamma),
        "sigma": float(sigma if sigma is not None else DEFAULT_GUIDANCE_SIGMA),
        "apply_every": max(1, int(apply_every)),
        "apply_during_sampling": apply_during_sampling,
        "blur_sigma_px": max(0.0, float(blur_sigma_px)),
        "blur_sigma_px_final": (
            None if blur_sigma_px_final is None else max(0.0, float(blur_sigma_px_final))
        ),
        "blur_schedule_power": max(0.0, float(blur_schedule_power)),
        "tighten_final_steps": max(0, int(tighten_final_steps)),
        "recompute_eps": bool(recompute_eps),
    }


def _combine_seed(*values: int) -> int:
    """
    Deterministically combine integers into a 32-bit seed.

    We avoid Python's built-in hash() because it is salted per-process by default.
    """

    seed = 0
    for v in values:
        vv = int(v) & 0xFFFFFFFF
        seed ^= vv + 0x9E3779B9 + ((seed << 6) & 0xFFFFFFFF) + (seed >> 2)
        seed &= 0xFFFFFFFF
    return int(seed)


def _make_regular_grid_mask(
    nx: int,
    ny: int,
    grid_interval: int,
    init_x: int,
    init_y: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a boolean observation mask on an (nx, ny) HR grid using a regular lattice:
      mask[x, y] = True for x = init_x + k*grid_interval, y = init_y + l*grid_interval.

    For backward compatibility with the existing SRDA eval path (which uses 128×65 HR but
    drops the last y-column for ML), we keep the final y-column (index ny-1) always unobserved.
    """

    if grid_interval <= 0:
        raise ValueError("grid_interval must be >= 1")
    if ny < 2:
        raise ValueError("ny must be >= 2")

    dev = device if device is not None else torch.device("cpu")
    init_x = int(init_x) % int(grid_interval)
    init_y = int(init_y) % int(grid_interval)

    mask_core = torch.zeros((nx, ny - 1), dtype=torch.bool, device=dev)
    mask_core[init_x::grid_interval, init_y::grid_interval] = True

    mask = torch.zeros((nx, ny), dtype=torch.bool, device=dev)
    mask[:, :-1] = mask_core
    return mask


def _generate_sensor_shift_hr_observations(
    hr_omegas: torch.Tensor,
    *,
    uhr_seed: int,
    obs_noise_seed: int,
    cond_grid_interval: int,
    guidance_grid_interval: int,
    cond_obs_noise_sigma: float,
    guidance_obs_noise_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate two HR observation time series (NaN-sparse) for:
      - conditioning ("cond"): what the denoiser sees after preprocessing/fill
      - guidance ("guid"): what defines the likelihood mask/target for obs guidance

    Design goals:
      - Deterministic per (uhr_seed, time_idx) and independent of evaluation order.
      - Conditioning observations depend only on conditioning sensor settings.
      - When possible, the guidance mask is aligned to *contain* the conditioning mask
        (e.g. cond=8, guid=4), and overlapping sensor values are identical.
    """

    if hr_omegas.ndim != 3:
        raise ValueError("hr_omegas must have shape (T, nx, ny)")

    n_time, nx, ny = hr_omegas.shape
    cond_interval = int(cond_grid_interval)
    guid_interval = int(guidance_grid_interval)
    if cond_interval <= 0 or guid_interval <= 0:
        raise ValueError("cond_grid_interval and guidance_grid_interval must be >= 1")

    cond_sigma = max(0.0, float(cond_obs_noise_sigma))
    guid_sigma = max(0.0, float(guidance_obs_noise_sigma))
    base_seed = int(obs_noise_seed)
    uhr_seed_i = int(uhr_seed)

    obs_cond = torch.full_like(hr_omegas, float("nan"))
    obs_guid = torch.full_like(hr_omegas, float("nan"))

    for t in range(n_time):
        # Conditioning mask + noise: depends only on conditioning settings.
        cond_seed = _combine_seed(base_seed, uhr_seed_i, int(t), cond_interval, 0xC0DE)
        rng_cond = np.random.RandomState(cond_seed)
        cond_init_x = int(rng_cond.randint(0, cond_interval))
        cond_init_y = int(rng_cond.randint(0, cond_interval))
        mask_cond = _make_regular_grid_mask(nx, ny, cond_interval, cond_init_x, cond_init_y)

        obs_cond_t = torch.full((nx, ny), float("nan"), dtype=hr_omegas.dtype)
        truth_t = hr_omegas[t]
        obs_cond_t[mask_cond] = truth_t[mask_cond]
        if cond_sigma > 0 and torch.any(mask_cond):
            z = rng_cond.normal(loc=0.0, scale=cond_sigma, size=(nx, ny))
            noise = torch.from_numpy(z).to(dtype=obs_cond_t.dtype)
            obs_cond_t[mask_cond] = obs_cond_t[mask_cond] + noise[mask_cond]

        # Guidance mask: when possible, align to contain the conditioning mask.
        guid_seed = _combine_seed(base_seed, uhr_seed_i, int(t), guid_interval, 0xBEEF)
        rng_guid = np.random.RandomState(guid_seed)
        if guid_interval <= cond_interval and (cond_interval % guid_interval == 0):
            guid_init_x = cond_init_x % guid_interval
            guid_init_y = cond_init_y % guid_interval
        else:
            guid_init_x = int(rng_guid.randint(0, guid_interval))
            guid_init_y = int(rng_guid.randint(0, guid_interval))

        mask_guid = _make_regular_grid_mask(nx, ny, guid_interval, guid_init_x, guid_init_y)
        obs_guid_t = torch.full((nx, ny), float("nan"), dtype=hr_omegas.dtype)
        obs_guid_t[mask_guid] = truth_t[mask_guid]

        overlap = mask_guid & mask_cond
        # By default, shared sensors correspond to the same physical instruments, so their
        # measurement values (including injected noise) should match between conditioning and
        # guidance. The only exception is when the user explicitly asks for different guidance
        # noise on the same sampling grid (cond==guid): then we treat guidance as an independent
        # observation draw on the same mask.
        copy_overlap = not (guid_interval == cond_interval and guid_sigma != cond_sigma)
        if copy_overlap and torch.any(overlap):
            obs_guid_t[overlap] = obs_cond_t[overlap]

        guid_only = mask_guid if not copy_overlap else (mask_guid & (~overlap))
        if guid_sigma > 0 and torch.any(guid_only):
            z = rng_guid.normal(loc=0.0, scale=guid_sigma, size=(nx, ny))
            noise = torch.from_numpy(z).to(dtype=obs_guid_t.dtype)
            obs_guid_t[guid_only] = obs_guid_t[guid_only] + noise[guid_only]

        obs_cond[t] = obs_cond_t
        obs_guid[t] = obs_guid_t

    return obs_cond, obs_guid


def _extract_obs_raw_window(
    hr_obs: List[torch.Tensor],
    dataset: SrdaByDdpmDataset,
    *,
    assimilation_period: int,
) -> torch.Tensor:
    """
    Extract the same raw (NaN-sparse) obs window used for guidance mask construction,
    but without running `process_obs` / fill interpolation.

    Matches `make_preprocessed_obs_for_forecast_with_raw(...)[1]` semantics:
      normalize → drop_y_boundary → (clone) before filling.
    """

    obs = hr_obs[-(assimilation_period + 1) :]
    obs = obs[:: dataset.obs_time_interval]
    obs = torch.stack(obs, dim=0)
    obs = dataset.normalize(obs.to(dataset.dtype))
    obs = dataset.drop_y_boundary(obs)
    return obs


def _gaussian_blur_residual(residual: torch.Tensor, sigma_px: float) -> torch.Tensor:
    if sigma_px <= 0:
        return residual
    radius = max(1, int(3 * sigma_px))
    if radius <= 0:
        return residual
    device = residual.device
    dtype = residual.dtype
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma_px) ** 2)
    kernel = kernel / kernel.sum()
    kernel_col = kernel.view(1, 1, -1, 1)
    kernel_row = kernel.view(1, 1, 1, -1)
    channels = residual.shape[1]
    out = F.conv2d(
        residual,
        kernel_col.expand(channels, 1, -1, 1),
        padding=(radius, 0),
        groups=channels,
    )
    out = F.conv2d(
        out,
        kernel_row.expand(channels, 1, 1, -1),
        padding=(0, radius),
        groups=channels,
    )
    return out


def _apply_final_projection(sr: torch.Tensor, obs_guidance: dict) -> torch.Tensor:
    mask = obs_guidance["mask"].to(sr.device)
    target = obs_guidance["target"].to(sr.device)
    if mask.shape[0] != sr.shape[0]:
        mask = mask.expand(sr.shape[0], -1, -1, -1)
        target = target.expand(sr.shape[0], -1, -1, -1)

    mask_float = mask.to(sr.dtype)
    residual = (target - sr) * mask_float
    blur_sigma = obs_guidance.get("blur_sigma_px", 0.0)
    if blur_sigma > 0:
        residual = _gaussian_blur_residual(residual, blur_sigma)

    if obs_guidance["mode"] == "hard":
        updated = sr + residual
        return torch.where(mask, target, updated)

    alpha = obs_guidance["gamma"] / (
        obs_guidance["gamma"] + obs_guidance["sigma"] ** 2
    )
    return sr + alpha * residual


def evaluate_srda_dm_and_generate_obs(
    i_seed_start,
    i_seed_end,
    root_dir,
    experiment_name,
    test_name,
    device,
    batch_size=30,
    config_path: Optional[str] = None,
    config: Optional[dict] = None,
    processed_base_dir: Optional[str] = None,
    run_suffix: Optional[str] = None,
    obs_guidance_space: str = "auto",
    obs_guidance_mode: str = "off",
    obs_guidance_gamma: float = 1.0,
    obs_guidance_sigma: Optional[float] = None,
    obs_guidance_every: int = 1,
    obs_guidance_blur_sigma_px: float = 0.0,
    obs_guidance_recompute_eps: bool = False,
    obs_guidance_tighten_final_steps: int = 0,
    obs_guidance_blur_sigma_px_final: Optional[float] = None,
    obs_guidance_blur_schedule_power: float = 1.0,
    cond_grid_interval: Optional[int] = None,
    guidance_grid_interval: Optional[int] = None,
    cond_obs_noise_sigma: Optional[float] = None,
    guidance_obs_noise_sigma: Optional[float] = None,
    obs_noise_seed: Optional[int] = None,
    sensor_scenario: str = "legacy",
    sensor_seed: Optional[int] = None,
    sensor_obs_noise_sigma: Optional[float] = None,
    sensor_grid_interval: Optional[int] = None,
    sensor_num_sensors: Optional[int] = None,
    timing: bool = False,
    timing_out_dir: Optional[str] = None,
    timing_warmup_cycles: int = 0,
    timing_skip_save_outputs: bool = False,
):
    BATCH_SIZE = batch_size
    timing_enabled = bool(timing)
    skip_save_outputs = bool(timing_enabled and timing_skip_save_outputs)
    if config is not None and config_path is not None:
        raise ValueError("Pass either config dict or config_path, not both.")
    if config is None:
        inferred_path = config_path or f"{root_dir}/python/configs/srda/{experiment_name}/{test_name}.yaml"
        with open(inferred_path, "r") as file:
            CONFIG = yaml.safe_load(file)
    else:
        CONFIG = deepcopy(config)

    try:
        latent_model_config_path = root_dir + CONFIG["path"]["latent_model"]
        with open(latent_model_config_path, "r") as file:
            LATENT_CONFIG = yaml.safe_load(file)
    except:
        LATENT_CONFIG = None

    LR_CFD_CONFIG = {
        "nx": LR_NX,
        "ny": LR_NY,
        "coeff_linear_drag": COEFF_LINEAR_DRAG,
        "coeff_diffusion": LR_COEFF_DIFFUSION,
        "order_diffusion": ORDER_DIFFUSION,
        "beta": BETA,
        "device": device,
        "dt": LR_DT,
        "nt": LR_NT,
    }

    INDEX_CONFIG = {
        "assimilation_period": ASSIMILATION_PERIOD,
        "forecast_span": FORECAST_SPAN,
        "n_ens": 1,
        "lr_nx": LR_NX,
        "lr_ny": LR_NY,
        "hr_nx": HR_NX,
        "hr_ny": HR_NY,
        "device": device,
    }

    set_seeds(555, use_deterministic=True)

    init_hr_omega = get_initial_hr_omega(
        nx=HR_NX,
        ny=HR_NY,
        num_simulations=NUM_SIMULATIONS,
        y0=Y0,
        sigma=SIGMA,
        tau0=TAU0,
        pertub_noise=PERTUB_NOISE,
        u0=U0,
    )
    assert init_hr_omega.shape == (1, HR_NX, HR_NY)

    _, lr_forcing = calc_jet_forcing(
        nx=LR_NX,
        ny=LR_NY,
        ne=1,
        y0=Y0,
        sigma=SIGMA,
        tau0=TAU0,
    )
    assert lr_forcing.shape == (1, LR_NX, LR_NY)

    low_pass_filter = LowPassFilter(
        nx_lr=LR_NX, ny_lr=LR_NY, nx_hr=HR_NX, ny_hr=HR_NY, device=device
    )

    if LATENT_CONFIG:
        latent_model = load_latent_model(
            config=LATENT_CONFIG, root_dir=root_dir, device=device
        )
    else:
        latent_model = None

    diffusion = Model.create_model(opt=CONFIG, latent_model=latent_model, device=device)
    diffusion.set_new_noise_schedule(
        CONFIG["diffusion_model"]["beta_schedule"]["train"]
    )
    weights_path = f"{root_dir}{CONFIG['path']['model']}/weight_diffusion.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Diffusion weights not found: {weights_path}")
    logger.info(f"Loading diffusion weights: {weights_path}")
    diffusion.netG.load_state_dict(torch.load(weights_path, map_location="cpu"))
    diffusion.netG.to(device)
    _ = diffusion.netG.eval()

    tmp_files1 = f"{root_dir}{CONFIG['datasets']['data_dir']}/seed00000/seed00000_start00_end08_hr_omega_00.npy"
    tmp_files2 = f"{root_dir}{CONFIG['datasets']['data_dir']}/seed00000/seed00000_start00_end08_hr_omega_01.npy"
    _config = {"train_files": [tmp_files1], "valid_files": [tmp_files2]}
    my_dataset = make_dataloaders(
        **_config,
        **CONFIG["datasets"],
    )["valid"].dataset

    base_grid_interval = int(CONFIG.get("datasets", {}).get("obs_grid_interval", GRID_INTERVAL))
    base_obs_noise_std = float(CONFIG.get("datasets", {}).get("obs_noise_std", 0.0))
    sensor_scenario = str(sensor_scenario or "legacy").lower()
    sensor_scenario_requested = sensor_scenario != "legacy"
    sensor_shift_requested = any(
        v is not None
        for v in (
            cond_grid_interval,
            guidance_grid_interval,
            cond_obs_noise_sigma,
            guidance_obs_noise_sigma,
            obs_noise_seed,
        )
    )
    if sensor_scenario_requested and any(
        v is not None
        for v in (
            cond_grid_interval,
            guidance_grid_interval,
            cond_obs_noise_sigma,
            guidance_obs_noise_sigma,
        )
    ):
        raise ValueError(
            "sensor_scenario != 'legacy' cannot be combined with cond/guid sensor-shift knobs "
            "(cond_grid_interval/guidance_grid_interval/cond_obs_noise_sigma/guidance_obs_noise_sigma). "
            "Use sensor_scenario arguments instead."
        )

    if sensor_scenario_requested:
        # Defaults: match the YAML unless explicitly overridden.
        sensor_seed_eff = (
            int(sensor_seed)
            if sensor_seed is not None
            else (int(obs_noise_seed) if obs_noise_seed is not None else OBS_SRDA_SEED)
        )
        sensor_obs_noise_sigma_eff = (
            float(sensor_obs_noise_sigma)
            if sensor_obs_noise_sigma is not None
            else float(base_obs_noise_std)
        )
        if sensor_obs_noise_sigma_eff < 0:
            raise ValueError("sensor_obs_noise_sigma must be >= 0.")
        sensor_grid_interval_eff = (
            int(sensor_grid_interval)
            if sensor_grid_interval is not None
            else int(base_grid_interval)
        )
        if sensor_grid_interval_eff <= 0:
            raise ValueError("sensor_grid_interval must be >= 1.")

        if sensor_scenario == "random_uniform_fixed":
            if sensor_num_sensors is None or int(sensor_num_sensors) <= 0:
                raise ValueError(
                    "sensor_num_sensors must be set and >0 for sensor_scenario=random_uniform_fixed."
                )

        if not run_suffix:
            suffix_parts = []
            if sensor_scenario == "regular_grid_fixed":
                suffix_parts.append(f"sensregfix_ogi{sensor_grid_interval_eff:02d}")
            elif sensor_scenario == "random_uniform_fixed":
                suffix_parts.append(f"sensrandfix_Ns{int(sensor_num_sensors)}")
            else:
                raise ValueError(f"Unknown sensor_scenario: {sensor_scenario}")
            suffix_parts.append(f"sn{str(sensor_obs_noise_sigma_eff).replace('.','p')}")
            suffix_parts.append(f"sseed{int(sensor_seed_eff)}")
            run_suffix = "_".join(suffix_parts)

    # Sensor scenarios override the legacy "sensor shift" pathway.
    sensor_shift_requested = bool(sensor_shift_requested and not sensor_scenario_requested)

    if sensor_shift_requested:
        cond_grid_interval = (
            int(cond_grid_interval) if cond_grid_interval is not None else base_grid_interval
        )
        guidance_grid_interval = (
            int(guidance_grid_interval)
            if guidance_grid_interval is not None
            else int(cond_grid_interval)
        )
        cond_obs_noise_sigma = (
            float(cond_obs_noise_sigma)
            if cond_obs_noise_sigma is not None
            else base_obs_noise_std
        )
        guidance_obs_noise_sigma = (
            float(guidance_obs_noise_sigma)
            if guidance_obs_noise_sigma is not None
            else float(cond_obs_noise_sigma)
        )
        obs_noise_seed = int(obs_noise_seed) if obs_noise_seed is not None else OBS_SRDA_SEED

        if cond_grid_interval <= 0 or guidance_grid_interval <= 0:
            raise ValueError("cond_grid_interval and guidance_grid_interval must be >= 1")
        if cond_obs_noise_sigma < 0 or guidance_obs_noise_sigma < 0:
            raise ValueError("cond_obs_noise_sigma and guidance_obs_noise_sigma must be >= 0")

        if not run_suffix:
            # Safety: never overwrite baseline outputs when sensor-shift knobs are enabled.
            suffix_parts = []
            if int(cond_grid_interval) == int(guidance_grid_interval):
                suffix_parts.append(f"ogi{int(cond_grid_interval):02d}")
            else:
                suffix_parts.append(f"condogi{int(cond_grid_interval):02d}")
                suffix_parts.append(f"guidogi{int(guidance_grid_interval):02d}")

            cond_sigma_label = str(float(cond_obs_noise_sigma)).replace(".", "p")
            guid_sigma_label = str(float(guidance_obs_noise_sigma)).replace(".", "p")
            if float(cond_obs_noise_sigma) == float(guidance_obs_noise_sigma):
                suffix_parts.append(f"n{cond_sigma_label}")
            else:
                suffix_parts.append(f"condn{cond_sigma_label}")
                suffix_parts.append(f"guidn{guid_sigma_label}")

            suffix_parts.append(f"obsseed{int(obs_noise_seed)}")
            run_suffix = "_".join(suffix_parts)
    guidance_mode = (obs_guidance_mode or "off").lower()
    if guidance_mode not in {"off", "hard", "soft"}:
        raise ValueError("obs_guidance_mode must be one of: off, hard, soft")
    sr_channels = CONFIG["diffusion_model"]["diffusion"]["channels"]

    guidance_space = (obs_guidance_space or "auto").lower()
    if guidance_space == "auto":
        # Backward-compatible default:
        # - Pixel-space diffusion (no latent model): apply guidance during sampling.
        # - Latent diffusion (VQ-VAE/LDM): apply one HR projection after decoding.
        guidance_space = "pixel" if latent_model is None else "post_hr"
    if guidance_space not in {"pixel", "post_hr"}:
        raise ValueError(
            "obs_guidance_space must be one of: auto, pixel, post_hr"
        )

    suffix_str = f"_{run_suffix}" if run_suffix else ""

    exp_output_root = resolve_experiment_output_root(
        root_dir=root_dir,
        experiment_name=experiment_name,
        processed_base_dir=processed_base_dir,
    )

    timing_recorder: Optional[TimingRecorder] = None
    if timing_enabled:
        try:
            val_schedule = CONFIG["diffusion_model"]["beta_schedule"]["val"]
        except Exception:
            val_schedule = {}
        respacing_cfg = (
            val_schedule.get("timestep_respacing")
            if isinstance(val_schedule, dict)
            else None
        )
        respacing_steps = _infer_sampling_steps(respacing_cfg)
        eta_cfg = None
        try:
            eta_cfg = CONFIG["diffusion_model"]["diffusion"].get("eta")
        except Exception:
            eta_cfg = None

        base_dir = (
            pathlib.Path(timing_out_dir)
            if timing_out_dir
            else (exp_output_root / "timing" / test_name)
        )
        run_id = f"{(run_suffix or 'nosuffix')}_seeds{i_seed_start:05d}-{i_seed_end:05d}"
        timing_run_dir = base_dir / run_id
        meta = RunTimingSummary(
            experiment_name=str(experiment_name),
            test_name=str(test_name),
            seeds_start=int(i_seed_start),
            seeds_end=int(i_seed_end),
            run_suffix=str(run_suffix or ""),
            respacing=str(respacing_steps) if respacing_steps else None,
            eta=str(eta_cfg) if eta_cfg is not None else None,
            batch_size=int(batch_size) if batch_size is not None else None,
            obs_guidance_mode=str(guidance_mode),
            obs_guidance_space=str(guidance_space),
            extra={
                "timing_skip_save_outputs": bool(skip_save_outputs),
            },
        )
        timing_recorder = TimingRecorder(
            out_dir=timing_run_dir,
            run_meta=meta,
            warmup_cycles=int(timing_warmup_cycles),
        )

    for i_seed_uhr in range(i_seed_start, i_seed_end + 1):
        seed_timer = CudaWallTimer(device=device, enabled=timing_enabled)
        seed_timer.start()
        seed_cycle_rows: List[CycleTimingRow] = []

        UHR_RESULT_DIR = (
            f"{root_dir}/data/ddpm/external/uhr_0050_simulations/seed{i_seed_uhr:05}"
        )
        data_dir = exp_output_root / "data" / test_name / f"{i_seed_uhr:05}"
        csv_dir = exp_output_root / "csv" / test_name / f"{i_seed_uhr:05}"
        fig_dir = exp_output_root / "fig" / test_name / f"{i_seed_uhr:05}"
        for _dir in (data_dir, csv_dir, fig_dir):
            os.makedirs(_dir, exist_ok=True)

        output_npz_file_path = (
            data_dir
            / f"UHR_seed_{i_seed_uhr:05}_og{GRID_INTERVAL:02}_SRDA_seed_{OBS_SRDA_SEED}{suffix_str}.npz"
        )

        uhr_omegas, hr_omegas = get_uhr_and_hr_omegas(
            result_dir=UHR_RESULT_DIR,
            uhr_nx=UHR_NX,
            uhr_ny=UHR_NY,
            nt=NUM_TIMES,
            hr_nx=HR_NX,
            hr_ny=HR_NY,
        )

        sensor_mask_series = None
        if sensor_scenario_requested:
            cfg = SensorScenarioConfig(
                scenario=str(sensor_scenario),
                sensor_seed=int(sensor_seed_eff),
                obs_noise_sigma=float(sensor_obs_noise_sigma_eff),
                assimilation_period=int(ASSIMILATION_PERIOD),
                grid_interval=int(sensor_grid_interval_eff),
                num_sensors=None if sensor_num_sensors is None else int(sensor_num_sensors),
            )
            hr_obsrvs_cond, sensor_mask_series, sensor_meta = (
                generate_sensor_scenario_hr_observations(
                    hr_omegas,
                    uhr_seed=i_seed_uhr,
                    cfg=cfg,
                )
            )
            hr_obsrvs_guid = hr_obsrvs_cond

            try:
                mask0 = sensor_mask_series[0]
                dens0 = (
                    mask0[:, :-1].to(torch.float32).mean().item()
                    if mask0.ndim == 2
                    else float("nan")
                )
                n0 = int(mask0[:, :-1].sum().item()) if mask0.ndim == 2 else 0
                msg = (
                    f"[Seed {i_seed_uhr:05d}] Sensor scenario: {sensor_scenario}, "
                    f"dens0={dens0:.4f}, n0={n0}, "
                    f"sseed={int(sensor_seed_eff)}, sn={float(sensor_obs_noise_sigma_eff)}"
                )
                logger.info(msg)
            except Exception:
                pass
        elif sensor_shift_requested:
            hr_obsrvs_cond, hr_obsrvs_guid = _generate_sensor_shift_hr_observations(
                hr_omegas,
                uhr_seed=i_seed_uhr,
                obs_noise_seed=int(obs_noise_seed),
                cond_grid_interval=int(cond_grid_interval),
                guidance_grid_interval=int(guidance_grid_interval),
                cond_obs_noise_sigma=float(cond_obs_noise_sigma),
                guidance_obs_noise_sigma=float(guidance_obs_noise_sigma),
            )
            try:
                cond_density = (
                    torch.isfinite(hr_obsrvs_cond[0][:, :-1]).to(torch.float32).mean().item()
                )
                guid_density = (
                    torch.isfinite(hr_obsrvs_guid[0][:, :-1]).to(torch.float32).mean().item()
                )
            except Exception:
                cond_density = float("nan")
                guid_density = float("nan")
            logger.info(
                f"[Seed {i_seed_uhr:05d}] Sensor shift: "
                f"cond_ogi={int(cond_grid_interval):02d}, guid_ogi={int(guidance_grid_interval):02d}, "
                f"cond_sigma={float(cond_obs_noise_sigma)}, guid_sigma={float(guidance_obs_noise_sigma)}, "
                f"obs_seed={int(obs_noise_seed)}, "
                f"dens_cond={cond_density:.4f}, dens_guid={guid_density:.4f}"
            )
        else:
            hr_obsrvs_cond = get_observation_with_noise(
                hr_omegas[None, ...],
                my_dataset,
                **INDEX_CONFIG,  # add ens channel (dummy channel)
            ).squeeze()
            hr_obsrvs_guid = hr_obsrvs_cond

        # Always materialize explicit boolean mask series for visualization/debugging.
        # Shape: (NUM_TIMES, HR_NX, HR_NY). Last y-column remains unobserved by construction.
        try:
            if sensor_mask_series is not None:
                obs_mask_cond_series = sensor_mask_series.to(dtype=torch.bool).detach().cpu()
                obs_mask_guid_series = obs_mask_cond_series
            else:
                obs_mask_cond_series = torch.isfinite(hr_obsrvs_cond).to(torch.bool).detach().cpu()
                obs_mask_guid_series = torch.isfinite(hr_obsrvs_guid).to(torch.bool).detach().cpu()
        except Exception:
            obs_mask_cond_series = None
            obs_mask_guid_series = None

        assert uhr_omegas.shape == (NUM_TIMES, UHR_NX, UHR_NY)
        assert hr_omegas.shape == hr_obsrvs_cond.shape == (NUM_TIMES, HR_NX, HR_NY)
        assert hr_obsrvs_guid.shape == (NUM_TIMES, HR_NX, HR_NY)

        last_t0 = T0
        last_hr_omega0 = init_hr_omega

        hr_obs, hr_obs_guid, sr_forecast = [], [], []
        all_lr_forecasts = []
        obs_mae_mean = torch.full((NUM_TIMES,), torch.nan, dtype=torch.float32)
        obs_mae_ens = torch.full((BATCH_SIZE, NUM_TIMES), torch.nan, dtype=torch.float32)

        use_lr = latent_model is not None
        seed_failed = False
        for i_cycle in tqdm(range(NUM_TIMES)):
            if i_cycle % ASSIMILATION_PERIOD == 0:
                o_cond = hr_obsrvs_cond[i_cycle]
                o_guid = hr_obsrvs_guid[i_cycle]
                hr_obs.append(o_cond)
                hr_obs_guid.append(o_guid)
            else:
                o_cond = hr_obsrvs_cond[i_cycle]
                hr_obs.append(torch.full_like(o_cond, torch.nan))
                hr_obs_guid.append(torch.full_like(o_cond, torch.nan))

            if i_cycle > 0 and i_cycle % ASSIMILATION_PERIOD == 0:
                cycle_timer = CudaWallTimer(device=device, enabled=timing_enabled)
                lr_timer = CudaWallTimer(device=device, enabled=timing_enabled)
                sample_timer = CudaWallTimer(device=device, enabled=timing_enabled)
                cycle_timer.start()
                lr_timer.start()
                try:
                    lr_forecasts = initialize_and_itegrate_srda_cfd_model_for_forecast(
                        num_integrate_steps=ASSIMILATION_PERIOD + FORECAST_SPAN,
                        last_t0=last_t0,
                        last_hr_omega0=last_hr_omega0,
                        lr_ens_forcing=lr_forcing,
                        cfd_config=LR_CFD_CONFIG,
                        low_pass_filter=low_pass_filter,
                    )
                except Exception as exc:
                    logger.error(
                        f"[Seed {i_seed_uhr}] LR CFD forecast failed with error: {exc}. Skipping this seed."
                    )
                    seed_failed = True
                    break
                lr_forecast_sec = lr_timer.stop()
                assert len(lr_forecasts) == ASSIMILATION_PERIOD + FORECAST_SPAN + 1
                lr = torch.stack(lr_forecasts, dim=0)  # stack along time
                lr = lr.squeeze()
                lr = lr[:: my_dataset.obs_time_interval]
                all_lr_forecasts.append(lr)

                x = make_preprocessed_lr_for_forecast(
                    lr_forecasts, my_dataset, use_lr=use_lr
                )
                # Conditioning obs preprocessing must respect the conditioning grid interval
                # (SrdaByDdpmDatasetScipy.fill_obs depends on dataset.obs_grid_interval).
                orig_interval = getattr(my_dataset, "obs_grid_interval", None)
                need_interval_override = (
                    sensor_shift_requested
                    and orig_interval is not None
                    and int(orig_interval) != int(cond_grid_interval)
                )
                if need_interval_override:
                    my_dataset.obs_grid_interval = int(cond_grid_interval)
                try:
                    o, o_raw = make_preprocessed_obs_for_forecast_with_raw(
                        hr_obs=hr_obs,
                        dataset=my_dataset,
                        assimilation_period=ASSIMILATION_PERIOD,
                        use_lr=use_lr,
                    )
                finally:
                    if need_interval_override and orig_interval is not None:
                        my_dataset.obs_grid_interval = orig_interval

                o_raw_guid = (
                    _extract_obs_raw_window(
                        hr_obs_guid,
                        my_dataset,
                        assimilation_period=ASSIMILATION_PERIOD,
                    )
                    if sensor_shift_requested
                    else o_raw
                )
                expected_cond_ch = int(CONFIG["diffusion_model"]["unet"]["in_channel"]) - int(
                    CONFIG["diffusion_model"]["diffusion"]["channels"]
                )
                actual_cond_ch = int(x.shape[0]) + int(o.shape[0])
                if expected_cond_ch != actual_cond_ch:
                    raise ValueError(
                        f"Conditioning channel mismatch: expected_cond_ch={expected_cond_ch}, "
                        f"but got x({int(x.shape[0])})+obs({int(o.shape[0])})={actual_cond_ch}."
                    )


                if use_lr:
                    lr_omegas = torch.broadcast_to(
                        x, size=(BATCH_SIZE, x.shape[0], x.shape[1], x.shape[2])
                    ).to(my_dataset.dtype)
                    obs_omegas = torch.broadcast_to(
                        o,
                        size=(
                            BATCH_SIZE,
                            o.shape[0],
                            o.shape[1],
                            o.shape[2],
                        ),
                    ).to(my_dataset.dtype)
                else:
                    lr_omega_interpotated = torch.broadcast_to(
                        x, size=(BATCH_SIZE, x.shape[0], x.shape[1], x.shape[2])
                    ).to(my_dataset.dtype)
                    obs_omegas = torch.broadcast_to(
                        o, size=(BATCH_SIZE, o.shape[0], o.shape[1], o.shape[2])
                    ).to(my_dataset.dtype)

                obs_guidance_cfg = None
                if guidance_mode != "off":
                    apply_during_sampling = guidance_space == "pixel"
                    obs_guidance_cfg = _prepare_obs_guidance(
                        mode=guidance_mode,
                        obs_raw=o_raw_guid,
                        sr_channels=sr_channels,
                        dtype=my_dataset.dtype,
                        gamma=obs_guidance_gamma,
                        sigma=obs_guidance_sigma,
                        apply_every=obs_guidance_every,
                        apply_during_sampling=apply_during_sampling,
                        blur_sigma_px=obs_guidance_blur_sigma_px,
                        recompute_eps=obs_guidance_recompute_eps,
                        tighten_final_steps=obs_guidance_tighten_final_steps,
                        blur_sigma_px_final=obs_guidance_blur_sigma_px_final,
                        blur_schedule_power=obs_guidance_blur_schedule_power,
                    )
                    if obs_guidance_cfg is not None:
                        obs_guidance_cfg["space"] = guidance_space

                sample_timer.start()
                with torch.no_grad():
                    if latent_model:
                        inp = torch.cat((lr_omegas, obs_omegas), dim=1)
                        diffusion.feed_data(
                            {
                                "HR": torch.full((1, 5, 128, 64), torch.nan),
                                "SR": inp,
                                "True": torch.full((1, 5, 128, 64), torch.nan),
                            }
                        )
                    else:
                        inp = torch.cat((lr_omega_interpotated, obs_omegas), dim=1)
                        diffusion.feed_data(
                            {"HR": torch.full((1, 128, 64), torch.nan), "SR": inp}
                        )
                    diffusion.test(
                        continous=False,
                        hide_progress_bar=timing_enabled,
                        obs_guidance=(
                            obs_guidance_cfg
                            if obs_guidance_cfg
                            and obs_guidance_cfg["apply_during_sampling"]
                            else None
                        ),
                    )
                    visuals = diffusion.get_current_visuals()
                    diffusion_processes = visuals["SR"]
                    sr = diffusion_processes[-BATCH_SIZE:]
                    sr = sr.detach().cpu()
                    if (
                        obs_guidance_cfg
                        and not obs_guidance_cfg["apply_during_sampling"]
                    ):
                        sr = _apply_final_projection(sr, obs_guidance_cfg)

                sampling_sec = sample_timer.stop()
                sr = make_invprocessed_sr_for_forecast(sr, my_dataset)

                obs_curr = hr_obsrvs_guid[i_cycle].to(torch.float32)
                analysis = sr[:, 0].to(torch.float32)
                mask2d = torch.isfinite(obs_curr)
                if torch.any(mask2d):
                    obs_filled = torch.nan_to_num(obs_curr, nan=0.0)
                    diff = (analysis - obs_filled[None, ...]).abs()
                    mask_f = mask2d.to(diff.dtype)
                    denom = mask_f.sum()
                    if denom > 0:
                        mae_ens = (diff * mask_f).sum(dim=(1, 2)) / denom
                        obs_mae_ens[:, i_cycle] = mae_ens
                        obs_mae_mean[i_cycle] = mae_ens.mean()

                last_hr_omega0 = sr[0, 0]  # torch.mean(sr[:, 0, ...], axis=0)
                last_hr_omega0 = last_hr_omega0[None, ...]  # add ensemble dim
                last_t0 += ASSIMILATION_PERIOD * LR_DT * LR_NT

                # The indices between 0 to ASSIMILATION_PERIOD are past
                # So NaN values are substituted for the forecast.
                if len(sr_forecast) == 0:
                    dummy = torch.full(
                        size=(
                            BATCH_SIZE,
                            ASSIMILATION_PERIOD,
                        )
                        + sr.shape[2:],
                        fill_value=torch.nan,
                        dtype=sr.dtype,
                    )
                    sr_forecast.append(dummy)
                sr_forecast.append(sr[:, 0:4])

                cycle_total_sec = cycle_timer.stop()
                if timing_recorder is not None:
                    try:
                        val_schedule = CONFIG["diffusion_model"]["beta_schedule"]["val"]
                    except Exception:
                        val_schedule = {}
                    respacing_cfg = (
                        val_schedule.get("timestep_respacing")
                        if isinstance(val_schedule, dict)
                        else None
                    )
                    sampling_steps = _infer_sampling_steps(respacing_cfg)
                    ms_per_step = (
                        (sampling_sec / sampling_steps) * 1000.0
                        if sampling_steps > 0
                        else float("nan")
                    )
                    row = CycleTimingRow(
                        seed=int(i_seed_uhr),
                        i_cycle=int(i_cycle),
                        cycle_index=int(i_cycle // ASSIMILATION_PERIOD),
                        sampling_steps=int(sampling_steps),
                        cycle_total_sec=float(cycle_total_sec),
                        sampling_sec=float(sampling_sec),
                        sampling_ms_per_step=float(ms_per_step),
                        lr_forecast_sec=float(lr_forecast_sec),
                    )
                    seed_cycle_rows.append(row)
                    timing_recorder.add_cycle_row(row)

        if seed_failed:
            continue

        # Stack along time dim
        hr_obs = torch.stack(hr_obs, dim=0)
        hr_obs_guid = torch.stack(hr_obs_guid, dim=0)
        sr_forecast = torch.cat(sr_forecast, dim=1)

        assert hr_obs.shape == hr_omegas.shape == (NUM_TIMES, HR_NX, HR_NY)
        assert hr_obs_guid.shape == (NUM_TIMES, HR_NX, HR_NY)
        assert sr_forecast.shape == (BATCH_SIZE, NUM_TIMES, HR_NX, HR_NY)

        diag_cols = [
            np.arange(NUM_TIMES, dtype=np.int64),
            obs_mae_mean.detach().cpu().numpy(),
        ]
        diag_header = [
            "Time",
            "obs_mae_mean",
        ]
        diag_arr = np.stack(diag_cols, axis=1)
        obs_diag_csv_path = csv_dir / f"obs_diag_seed_{i_seed_uhr:05}{suffix_str}.csv"
        if not skip_save_outputs:
            np.savetxt(
                obs_diag_csv_path,
                diag_arr,
                delimiter=",",
                header=",".join(diag_header),
                comments="",
            )
            save_payload = {
                "hr_obs": hr_obs,
                "dm_forecast": sr_forecast,
                "lr_forecasts": torch.stack(all_lr_forecasts, dim=0),
                "obs_mae_mean": obs_mae_mean.detach().cpu().numpy(),
                "obs_mae_ens": obs_mae_ens.detach().cpu().numpy(),
            }
            if sensor_shift_requested:
                save_payload["hr_obs_cond"] = hr_obs
                save_payload["hr_obs_guid"] = hr_obs_guid
                save_payload["cond_grid_interval"] = int(cond_grid_interval)
                save_payload["guidance_grid_interval"] = int(guidance_grid_interval)
                save_payload["cond_obs_noise_sigma"] = float(cond_obs_noise_sigma)
                save_payload["guidance_obs_noise_sigma"] = float(guidance_obs_noise_sigma)
                save_payload["obs_noise_seed"] = int(obs_noise_seed)
            if sensor_scenario_requested:
                save_payload["hr_obs_cond"] = hr_obs
                save_payload["hr_obs_guid"] = hr_obs_guid
                save_payload["sensor_scenario"] = str(sensor_scenario)
                save_payload["sensor_seed"] = int(sensor_seed_eff)
                save_payload["sensor_obs_noise_sigma"] = float(sensor_obs_noise_sigma_eff)
                save_payload["sensor_grid_interval"] = int(sensor_grid_interval_eff)
                if sensor_num_sensors is not None:
                    save_payload["sensor_num_sensors"] = int(sensor_num_sensors)
                if sensor_mask_series is not None:
                    save_payload["sensor_mask_series"] = sensor_mask_series
            if obs_mask_cond_series is not None:
                save_payload["obs_mask_cond_series"] = obs_mask_cond_series
            if obs_mask_guid_series is not None:
                save_payload["obs_mask_guid_series"] = obs_mask_guid_series

            np.savez(str(output_npz_file_path), **save_payload)

        if timing_recorder is not None:
            seed_total_sec = seed_timer.stop()
            timing_recorder.add_seed_summary(
                seed=int(i_seed_uhr),
                total_sec=float(seed_total_sec),
                rows_for_seed=seed_cycle_rows,
            )

    if timing_recorder is not None:
        timing_recorder.write()


