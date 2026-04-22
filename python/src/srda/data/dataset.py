import copy
import os
import random
import re
import sys
from logging import getLogger
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger()


def generate_is_obs_and_obs_matrix(
    *,
    nx: int,
    ny: int,
    init_index_x: int,
    init_index_y: int,
    interval_x: int,
    interval_y: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    assert 0 <= init_index_x <= interval_x - 1
    assert 0 <= init_index_y <= interval_y - 1

    is_obs = torch.zeros(nx, ny, dtype=dtype)
    is_obs[init_index_x::interval_x, init_index_y::interval_y] = 1.0

    obs_indices = is_obs.reshape(-1)
    obs_indices = torch.where(obs_indices == 1.0)[0]

    num_obs = len(obs_indices)

    obs_matrix = torch.zeros(num_obs, nx * ny, dtype=dtype)

    for i, j in enumerate(obs_indices):
        obs_matrix[i, j] = 1.0

    p = 100 * torch.sum(obs_matrix).item() / (nx * ny)
    logger.debug(f"observation prob = {p} [%]")

    return is_obs.to(device), obs_matrix.to(device)


class SrdaDataset(Dataset):
    def __init__(
        self,
        *,
        hr_file_paths: list[str],
        omega_max: float,
        omega_min: float,
        scale_factor: int,
        obs_time_interval: int,
        obs_grid_interval: int,
        obs_noise_std: float,
        use_observation: bool,
        beta_dist_alpha: float,
        beta_dist_beta: float,
        use_mixup: bool,
        mixup_num: int,
        fill_missing_value: bool,
        missing_value: float,
        nx: int,
        ny: int,
        min_start_time_index: int,
        max_start_time_index: int,
        dtype: torch.types = torch.float32,
        **kwargs,
    ):
        self.hr_file_paths = hr_file_paths
        self.omega_max = omega_max
        self.omega_min = omega_min
        self.scale_factor = scale_factor
        self.obs_time_interval = obs_time_interval
        self.obs_grid_interval = obs_grid_interval
        self.obs_noise_std = obs_noise_std
        self.use_observation = use_observation
        self.beta_dist_alpha = beta_dist_alpha
        self.beta_dist_beta = beta_dist_beta
        self.use_mixup = use_mixup
        self.mixup_num = mixup_num
        self.fill_missing_value = fill_missing_value
        self.missing_value = missing_value
        self.nx = nx
        self.ny = ny
        self.min_start_time_index = min_start_time_index
        self.max_start_time_index = max_start_time_index
        self.dtype = dtype

        assert self.nx % self.scale_factor == 0
        assert self.ny % self.scale_factor == 0

        self._load_all_lr_data_at_init_time()

        self.is_obses = []
        self.obs_matrices = []
        ratio_mean = []

        for init_x in tqdm(range(self.obs_grid_interval)):
            for init_y in range(self.obs_grid_interval):
                is_obs, obs_mat = generate_is_obs_and_obs_matrix(
                    nx=self.nx,
                    ny=self.ny,
                    init_index_x=init_x,
                    init_index_y=init_y,
                    interval_x=self.obs_grid_interval,
                    interval_y=self.obs_grid_interval,
                    dtype=self.dtype,
                )
                self.is_obses.append(is_obs)
                self.obs_matrices.append(obs_mat)
                ratio_mean.append(torch.mean(is_obs).item())
        ratio_mean = sum(ratio_mean) / len(ratio_mean)
        logger.debug(
            f"Observation interval = {self.obs_grid_interval}, Observation grid ratio = {ratio_mean}"
        )

    def __len__(self) -> int:
        return len(self.hr_file_paths)

    def _get_lr_key(self, lr_path: str) -> str:
        return re.search(r"start\d+_end\d+", os.path.basename(lr_path)).group()

    def _load_all_lr_data_at_init_time(self) -> None:
        self.dict_all_lr_data_at_init_time = {}

        for hr_path in tqdm(self.hr_file_paths):
            lr_path = hr_path.replace("hr_omega", "lr_omega_no-noise")

            key = self._get_lr_key(lr_path)
            if key not in self.dict_all_lr_data_at_init_time:
                self.dict_all_lr_data_at_init_time[key] = []
            lr = torch.from_numpy(np.load(lr_path)).to(self.dtype)
            self.dict_all_lr_data_at_init_time[key].append(
                {"data": lr[0], "path": lr_path}
            )
            # `0` means the initial time

    def _load_numpy_data_as_tensor(self, path: str) -> torch.Tensor:
        return torch.from_numpy(np.load(path)).to(self.dtype)

    def normalize(self, data: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
        # change data range to [0, 1]
        data = (data - self.omega_min) / (self.omega_max - self.omega_min)
        # change data range to [-1, 1]
        return 2 * data - 1

    def drop_y_boundary(self, data: torch.tensor) -> torch.tensor:
        """
        change image size
        if image size is 128*65, it changes to 128*64
        if image size is 32*17, it changes to 32*16
        """
        _, nx, ny = data.shape
        assert (nx, ny) == (128, 65) or (nx, ny) == (32, 17)

        # data = F.interpolate(
        #     data[None, ...], size=(nx, ny - 1), mode="bilinear"
        # ).squeeze(0)
        data = data[:, :, :-1]
        return data

    def _load_hr(self, path: str) -> torch.Tensor:
        data = self._load_numpy_data_as_tensor(path)
        data = self.normalize(data)
        return self.drop_y_boundary(data)

    def _extract_observation_without_noise(
        self, hr_omega: torch.Tensor
    ) -> torch.Tensor:
        i = torch.randint(low=0, high=len(self.is_obses), size=(1,)).item()
        is_obs = self.is_obses[i]
        assert is_obs.shape == hr_omega.shape[1:]

        is_obs = torch.broadcast_to(is_obs, hr_omega.shape)
        logger.debug(f"index of is_obs = {i}")

        obs = torch.full_like(hr_omega, torch.nan)
        obs = torch.where(is_obs > 0, hr_omega, obs)
        return obs

    def fill_obs(self, obs):
        for i in range(obs.shape[0]):
            o = obs[i]
            o = pd.DataFrame(o)
            o = o.interpolate(
                axis=0,
                limit_direction="both",
                method="linear",
            )
            o = o.interpolate(axis=1, limit_direction="both", method="linear")
            obs[i] = torch.Tensor(np.array(o))
        return obs

    def process_obs(self, obs):
        if self.fill_missing_value:
            obs = self.fill_obs(obs)
        else:
            obs = torch.nan_to_num(obs, nan=self.missing_value)
        return obs

    def _make_obs(self, hr_omega: torch.Tensor) -> torch.tensor:
        obs = self._extract_observation_without_noise(hr_omega)
        if self.obs_noise_std > 0:
            noise = torch.normal(mean=0, std=self.obs_noise_std, size=obs.shape).to(
                self.dtype
            )
            noise = self.normalize(noise)
            obs = obs + noise
        obs = obs[:: self.obs_time_interval]
        n = obs.shape[0]
        # 現在までの観測値のみ抽出する
        # タイムステップがn=９だったら、前半の５つを抽出
        # タイムステップがn=3だったら、前半の２つを抽出
        obs = obs[: n // 2 + 1]
        return self.process_obs(obs)

    def _get_similar_source_lr_path(self, key: str, target_lr: torch.tensor) -> str:
        all_lrs = self.dict_all_lr_data_at_init_time[key]

        min_path, min_norm = None, torch.inf
        logger.debug(
            f"Source data for mixup is selected by {min(self.mixup_num, len(all_lrs))} images"
        )
        for i in random.sample(
            range(0, len(all_lrs)), min(self.mixup_num, len(all_lrs))
        ):
            data = all_lrs[i]["data"]
            path = all_lrs[i]["path"]
            assert data.shape == target_lr.shape

            norm = torch.mean((data - target_lr) ** 2)
            if 0 < norm < min_norm:
                min_norm = norm
                min_path = path
                logger.debug(f"norm = {norm}, path = {min_path}")
        return min_path

    def _load_lr(self, target_lr_path: str) -> torch.tensor:
        key = re.search(r"start\d+_end\d+", os.path.basename(target_lr_path)).group()
        # Pass target_lr at the initial time, which has the index of `0`
        target_lr = self._load_numpy_data_as_tensor(target_lr_path)
        logger.debug(f"target_lr={target_lr_path}")
        logger.debug(f"lr key = {key}")
        if self.use_mixup:
            source_lr_path = self._get_similar_source_lr_path(key, target_lr[0])
            if source_lr_path is None:
                # Fallback: if the candidate pool only contains identical fields, MSE=0 for all,
                # so no "similar but different" source exists. In that case, skip mixup.
                lr_omega = target_lr
                logger.debug("No suitable mixup source found; falling back to target LR.")
            else:
                source_lr = self._load_numpy_data_as_tensor(source_lr_path)
                logger.debug(f"source_lr={source_lr_path}")
                source_prob = np.random.beta(
                    a=self.beta_dist_alpha, b=self.beta_dist_beta, size=1
                )[0]
                logger.debug(f"similar source_prob = {source_prob}")
                lr_omega = source_prob * source_lr + (1 - source_prob) * target_lr
        else:
            lr_omega = target_lr
        lr_omega = self.normalize(lr_omega)

        lr_omega = self.drop_y_boundary(lr_omega)
        return lr_omega[:: self.obs_time_interval]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hr_path = self.hr_file_paths[idx]
        hr_omega = self._load_hr(hr_path)

        obs_omega = self._make_obs(hr_omega)
        n = hr_omega.shape[0]
        # 現在以降のHR値のみ抽出する
        # タイムステップがn=９だったら、後半の５つを抽出
        hr_omega = hr_omega[n // 2 :]

        lr_path = hr_path.replace("_hr_omega_", "_lr_omega_no-noise_")
        lr_omega = self._load_lr(lr_path)
        return hr_omega, lr_omega, obs_omega

    def inv_normalize(self, x):
        out = (x + 1.0) / 2.0
        out = out * (self.omega_max - self.omega_min) + self.omega_min
        return out


class SrdaByDdpmDataset(SrdaDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hr_path = self.hr_file_paths[idx]
        hr_omega = self._load_hr(hr_path)

        obs_omega = self._make_obs(hr_omega)
        assert obs_omega.shape == (2, 128, 64)
        n = hr_omega.shape[0]
        # 現在以降のHR値のみ抽出する
        # タイムステップがn=９だったら、後半の５つを抽出
        hr_omega = hr_omega[n // 2 :]

        lr_path = hr_path.replace("_hr_omega_", "_lr_omega_no-noise_")
        lr_omega = self._load_lr(lr_path)
        lr_omega_interpotated = F.interpolate(
            lr_omega[None, ...], scale_factor=self.scale_factor, mode="bicubic"
        ).squeeze(0)

        return hr_omega, lr_omega, lr_omega_interpotated, obs_omega


class SrdaByDdpmDatasetScipy(SrdaDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.grid_x = torch.linspace(-1, 1, self.nx).to(self.device)
        self.grid_y = torch.linspace(-1, 1, self.ny).to(self.device)
        mesh_y, mesh_x = torch.meshgrid(self.grid_x, self.grid_y)
        self.points = torch.stack([mesh_x, mesh_y], dim=-1).unsqueeze(0).to(self.dtype)

    def fill_obs(self, obs):
        n = obs.shape[0]
        return_list = []
        for i in range(n):
            o = obs[i]
            mask = torch.isfinite(o)
            if not torch.any(mask):
                raise ValueError("fill_obs received an obs slice with no observed pixels.")

            xs_idx = torch.unique(torch.where(mask)[0])
            ys_idx = torch.unique(torch.where(mask)[1])

            n_obs = int(mask.sum().item())
            cart_mask = torch.zeros_like(mask)
            cart_mask[xs_idx[:, None], ys_idx[None, :]] = True
            is_cartesian = (int(cart_mask.sum().item()) == n_obs) and torch.all(mask == cart_mask)

            if is_cartesian:
                _xs = self.grid_x[xs_idx]
                _ys = self.grid_y[ys_idx]

                o = o[..., None]
                obs_inter = F.grid_sample(
                    o[torch.where(torch.isfinite(o))]
                    .reshape(_xs.shape[0], _ys.shape[0], 1)
                    .permute(2, 0, 1)
                    .unsqueeze(0),
                    self.points,
                    mode="bicubic",
                    align_corners=False,
                    padding_mode="border",
                ).squeeze(1)
                return_list.append(obs_inter)
                continue

            # Irregular sensor masks (random / clustered / dropout) are not a full (xs × ys) grid.
            # Use SciPy griddata to interpolate onto the full grid.
            from scipy.interpolate import griddata  # local import to keep import-time light

            nx, ny = int(o.shape[0]), int(o.shape[1])
            pts = torch.stack(torch.where(mask), dim=1).detach().cpu().numpy().astype(np.float64)
            vals = o[mask].detach().cpu().numpy().astype(np.float64)
            grid_x, grid_y = np.mgrid[0:nx, 0:ny]

            if pts.shape[0] < 3:
                filled = griddata(pts, vals, (grid_x, grid_y), method="nearest")
            else:
                filled = griddata(pts, vals, (grid_x, grid_y), method="linear")
                if np.isnan(filled).any():
                    filled_nn = griddata(pts, vals, (grid_x, grid_y), method="nearest")
                    filled = np.where(np.isnan(filled), filled_nn, filled)

            obs_inter = torch.from_numpy(filled).to(self.dtype).unsqueeze(0)
            return_list.append(obs_inter)
        return torch.cat(return_list, dim=0)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hr_path = self.hr_file_paths[idx]
        hr_omega = self._load_hr(hr_path)

        obs_omega = self._make_obs(hr_omega)
        assert obs_omega.shape == (2, 128, 64)
        n = hr_omega.shape[0]
        # 現在以降のHR値のみ抽出する
        # タイムステップがn=９だったら、後半の５つを抽出
        hr_omega = hr_omega[n // 2 :]

        lr_path = hr_path.replace("_hr_omega_", "_lr_omega_no-noise_")
        lr_omega = self._load_lr(lr_path)
        lr_omega_interpotated = F.interpolate(
            lr_omega[None, ...], scale_factor=self.scale_factor, mode="bicubic"
        ).squeeze(0)

        return hr_omega, lr_omega, lr_omega_interpotated, obs_omega


class SrdaByLdmDataset(SrdaDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hr_path = self.hr_file_paths[idx]
        hr_omega = self._load_hr(hr_path)

        obs_omega = self._make_obs(hr_omega)
        assert obs_omega.shape == (2, 128, 64)
        obs_omega = F.max_pool2d(
            obs_omega[None, ...], kernel_size=self.scale_factor
        ).squeeze(0)
        assert obs_omega.shape == (2, 32, 16)
        n = hr_omega.shape[0]
        # 現在以降のHR値のみ抽出する
        # タイムステップがn=９だったら、後半の５つを抽出
        hr_omega = hr_omega[n // 2 :]

        lr_path = hr_path.replace("_hr_omega_", "_lr_omega_no-noise_")
        lr_omega = self._load_lr(lr_path)

        dummy_lr_omega_interpotated = 0

        return hr_omega, lr_omega, dummy_lr_omega_interpotated, obs_omega

    def max_pool(self, obs_omega):
        assert obs_omega.shape == (2, 128, 64)
        obs_omega = F.max_pool2d(
            obs_omega[None, ...], kernel_size=self.scale_factor
        ).squeeze(0)
        assert obs_omega.shape == (2, 32, 16)
        return obs_omega


class LatentDataset(SrdaDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        hr_path = self.hr_file_paths[idx]
        hr = self._load_hr(hr_path)
        n = torch.randint(0, hr.shape[0], size=(1,)).item()
        return hr[n]
