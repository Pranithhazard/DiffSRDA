from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import torch


def _combine_seed(*values: int) -> int:
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
    if grid_interval <= 0:
        raise ValueError("grid_interval must be >= 1")
    if ny < 2:
        raise ValueError("ny must be >= 2")

    dev = device if device is not None else torch.device("cpu")
    init_x = int(init_x) % int(grid_interval)
    init_y = int(init_y) % int(grid_interval)

    mask = torch.zeros((nx, ny), dtype=torch.bool, device=dev)
    mask[init_x::grid_interval, init_y::grid_interval] = True
    mask[:, -1] = False
    return mask


def make_random_uniform_mask(
    nx: int,
    ny: int,
    num_sensors: int,
    *,
    rng: np.random.RandomState,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if num_sensors <= 0:
        raise ValueError("num_sensors must be > 0")
    if ny < 2:
        raise ValueError("ny must be >= 2")

    total = int(nx) * int(ny - 1)
    if num_sensors > total:
        raise ValueError(f"num_sensors={num_sensors} exceeds total available points={total}.")

    idx = rng.choice(total, size=int(num_sensors), replace=False)
    xs = (idx // (ny - 1)).astype(np.int64)
    ys = (idx % (ny - 1)).astype(np.int64)

    dev = device if device is not None else torch.device("cpu")
    mask = torch.zeros((nx, ny), dtype=torch.bool, device=dev)
    mask[torch.from_numpy(xs).to(dev), torch.from_numpy(ys).to(dev)] = True
    mask[:, -1] = False
    return mask


@dataclass(frozen=True)
class SensorScenarioConfig:
    scenario: str
    sensor_seed: int
    obs_noise_sigma: float
    assimilation_period: int
    grid_interval: Optional[int] = None
    num_sensors: Optional[int] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def generate_sensor_scenario_hr_observations(
    hr_omegas: torch.Tensor,
    *,
    uhr_seed: int,
    cfg: SensorScenarioConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if hr_omegas.ndim != 3:
        raise ValueError("hr_omegas must have shape (T, nx, ny)")

    scenario = str(cfg.scenario or "legacy").lower()
    if scenario == "legacy":
        raise ValueError("generate_sensor_scenario_hr_observations does not implement 'legacy'.")

    n_time, nx, ny = hr_omegas.shape
    if int(cfg.assimilation_period) <= 0:
        raise ValueError("assimilation_period must be > 0")

    base_seed = int(cfg.sensor_seed)
    uhr_seed_i = int(uhr_seed)
    dev = hr_omegas.device

    if scenario == "regular_grid_fixed":
        ogi = int(cfg.grid_interval or 0)
        if ogi <= 0:
            raise ValueError("grid_interval must be set and >= 1 for regular_grid_fixed.")
        rng = np.random.RandomState(_combine_seed(base_seed, uhr_seed_i, ogi, 0x0C1D))
        init_x = int(rng.randint(0, ogi))
        init_y = int(rng.randint(0, ogi))
        base_mask = _make_regular_grid_mask(nx, ny, ogi, init_x, init_y, device=dev)
    elif scenario == "random_uniform_fixed":
        ns = int(cfg.num_sensors or 0)
        if ns <= 0:
            raise ValueError("num_sensors must be set and >0 for random_uniform_fixed.")
        rng = np.random.RandomState(_combine_seed(base_seed, uhr_seed_i, ns, 0xA11C))
        base_mask = make_random_uniform_mask(nx, ny, ns, rng=rng, device=dev)
    else:
        raise ValueError(f"Unknown sensor scenario: {scenario}")

    if not torch.any(base_mask[:, :-1]):
        raise ValueError("Base sensor mask has zero observed points.")

    mask_series = torch.zeros((n_time, nx, ny), dtype=torch.bool, device=dev)
    for t in range(n_time):
        mask_series[t] = base_mask

    obs = torch.full_like(hr_omegas, float("nan"))
    sigma = max(0.0, float(cfg.obs_noise_sigma))
    for t in range(n_time):
        mask_t = mask_series[t]
        truth_t = hr_omegas[t]
        obs_t = torch.full((nx, ny), float("nan"), dtype=hr_omegas.dtype, device=dev)
        obs_t[mask_t] = truth_t[mask_t]
        if sigma > 0.0 and torch.any(mask_t):
            noise_seed = _combine_seed(base_seed, uhr_seed_i, int(t), 0xA015E)
            rng_noise = np.random.RandomState(noise_seed)
            z = rng_noise.normal(loc=0.0, scale=sigma, size=(nx, ny))
            noise = torch.from_numpy(z).to(device=dev, dtype=obs_t.dtype)
            obs_t[mask_t] = obs_t[mask_t] + noise[mask_t]
        obs[t] = obs_t

    meta: dict[str, Any] = {
        "scenario": scenario,
        "sensor_seed": int(cfg.sensor_seed),
        "obs_noise_sigma": float(cfg.obs_noise_sigma),
        "uhr_seed": int(uhr_seed),
    }
    return obs, mask_series, meta
