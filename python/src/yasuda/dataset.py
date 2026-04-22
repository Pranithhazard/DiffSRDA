import copy
import functools
import glob
import itertools
import os
import random
import re
import sys
import typing
from collections import OrderedDict
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


logger = getLogger()


class Dataset2dGaussJetSameTimeStep(Dataset):
    def __init__(
        self,
        *,
        lr_paths: typing.List[str],
        hr_paths: typing.List[str],
        list_bias: typing.List[float],
        list_scale: typing.List[float],
        prob_observation: float,
        missing_value: float = float("nan"),
        use_clamp: bool = False,
        clamp_min: float = None,
        clamp_max: float = None,
        start_time_index: int = 0,
        sampling_freq: int = 1,
        end_time_index_observation: int = None,
        dtype: torch.dtype = torch.float32,
    ):
        assert len(list_bias) == len(list_scale) == 3
        assert 0 <= prob_observation < 1

        assert len(lr_paths) == len(hr_paths)
        for l, h in zip(lr_paths, hr_paths):
            assert h.split("seed")[-1] == l.split("seed")[-1], "Seed is different."
        logger.info(f"Len paths = {len(lr_paths)}")

        self.lr_paths = copy.deepcopy(lr_paths)
        self.hr_paths = copy.deepcopy(hr_paths)
        self.prob_observation = prob_observation
        self.missing_value = torch.tensor(missing_value, dtype=dtype)
        logger.info(
            f"Porb obs = {self.prob_observation}, missing value = {self.missing_value}"
        )

        self.use_clamp = use_clamp
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        logger.info(
            f"use_clamp = {self.use_clamp}, min = {self.clamp_min}, max = {self.clamp_max}"
        )

        self.start_time_index = start_time_index
        self.sampling_freq = sampling_freq
        self.end_time_index_observation = end_time_index_observation
        if end_time_index_observation == "None" or end_time_index_observation == "none":
            self.end_time_index_observation = None
        logger.info(
            f"start_time_index = {self.start_time_index}, sampling freq = {self.sampling_freq}, end_time_index_obs = {self.end_time_index_observation}"
        )

        self.dtype = dtype

        # Add time, y, and x dims to broadcast
        logger.info(f"Bias = {list_bias}, Scale = {list_scale}")
        self.bias = torch.tensor(list_bias, dtype=dtype)[None, :, None, None]
        self.scale = torch.tensor(list_scale, dtype=dtype)[None, :, None, None]

    def __len__(self) -> int:
        return len(self.lr_paths)

    def _read_np_ndarray(self, path: str) -> torch.Tensor:
        data = np.load(path)
        return torch.from_numpy(data).to(self.dtype)

    def _get_obs_location(self, hr: torch.Tensor) -> torch.Tensor:
        assert hr.ndim == 4  # time, channel, y and x dims
        if self.prob_observation == 0 or self.end_time_index_observation == 0:
            return torch.zeros_like(hr)

        p = [1 - self.prob_observation, self.prob_observation]

        is_obs = np.random.choice(
            [0, 1], p=p, size=hr.shape[-2] * hr.shape[-1], replace=True
        )

        is_obs = is_obs.reshape(hr.shape[-2], hr.shape[-1])  # y and x dims
        is_obs = is_obs[None, None, ...]  # add time and channel dims

        # This assures same locations for all times and all channels
        is_obs = np.tile(is_obs, reps=(hr.shape[0], hr.shape[1], 1, 1))

        # No observation after `end_time_index_observation`
        if self.end_time_index_observation is not None:
            is_obs[self.end_time_index_observation :] = 0.0

        return torch.from_numpy(is_obs).to(hr.dtype)

    def _preprocess(
        self, data: torch.Tensor, use_clipping: bool = False
    ) -> torch.Tensor:

        ret = (data - self.bias) / self.scale
        if use_clipping:
            return torch.clamp(ret, min=self.clamp_min, max=self.clamp_max)
        return ret

    def _sample(self, data: torch.Tensor) -> torch.Tensor:
        ret = data[self.start_time_index :, ...]
        return ret[:: self.sampling_freq, ...]

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        lr = self._read_np_ndarray(self.lr_paths[idx])
        hr = self._read_np_ndarray(self.hr_paths[idx])

        lr = self._preprocess(lr, use_clipping=True)
        hr = self._preprocess(hr, use_clipping=self.use_clamp)

        lr = self._sample(lr)
        hr = self._sample(hr)

        is_obs = self._get_obs_location(hr)
        obs = torch.where(is_obs == 1, hr, self.missing_value)

        return lr, obs, hr


def extract_ground_truth_and_input(
    *,
    hr_omega: torch.Tensor,
    hr_obsrv: torch.Tensor,
    lr_assim: torch.Tensor,
    lr_frcst: torch.Tensor,
    i_start: int,
    n_snaps: int,
    assim_period: int,
    missing_value: float = torch.nan,
):
    # Check ensemble and time dimensions
    assert (
        hr_omega.shape[:2]
        == hr_obsrv.shape[:2]
        == lr_assim.shape[:2]
        == lr_frcst.shape[:2]
    )
    assert hr_omega.shape[0] == 1  # ensemble dimension

    i_end = i_start + n_snaps

    gt = hr_omega[0, i_start:i_end]  # ground truth

    # Observed every `assim_period`, which is started at the index of `0`
    # So the starting index must be adjusted.
    offset = 0
    if i_start % assim_period != 0:
        offset = assim_period - i_start % assim_period

    obs = torch.full_like(gt, missing_value)
    obs[offset::assim_period] = hr_obsrv[0, i_start + offset : i_end : assim_period]

    # Last low-resolution input is from forecast data, which is before assimilation
    lr = lr_assim[0, i_start:i_end]
    lr[-1] = lr_frcst[0, i_end - 1]

    return gt, obs, lr


class DatasetVorticityAssimilation(Dataset):
    def __init__(
        self,
        *,
        data_dirs: typing.List[str],
        assimilation_period: int,
        n_snapshots: int,
        obs_noise_std: float,
        biases: typing.List[float],
        scales: typing.List[float],
        is_always_start_observed: bool = True,
        initial_discarded_period: int = 16,
        max_sequence_length: int = 97,
        seed: int = 42,
        missing_value: float = torch.nan,
        clamp_min: float = None,
        clamp_max: float = None,
        use_clipping: bool = False,
        use_obs: bool = True,
        is_output_only_last: bool = False,
        dtype: torch.dtype = torch.float32,
        input_sampling_interval: int = None,
    ):

        assert assimilation_period > 0
        assert n_snapshots > 0
        assert obs_noise_std >= 0
        assert initial_discarded_period >= 0

        if is_output_only_last:
            assert is_always_start_observed is True
            assert input_sampling_interval is None

        if input_sampling_interval is not None:
            assert assimilation_period >= input_sampling_interval
            assert assimilation_period % input_sampling_interval == 0

        self.data_dirs = copy.deepcopy(data_dirs)
        self.assim_period = assimilation_period
        self.n_snapshots = n_snapshots
        self.obs_noise_std = obs_noise_std
        self.missing_value = missing_value
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.use_clipping = use_clipping
        self.use_obs = use_obs
        self.dtype = dtype
        self.np_rng = np.random.default_rng(seed)
        self.is_output_only_last = is_output_only_last
        self.input_sampling_interval = input_sampling_interval

        # Add time, x, and y dims to broadcast
        self.biases = torch.Tensor(biases)[None, :, None, None]
        self.scales = torch.Tensor(scales)[None, :, None, None]
        logger.info(f"biases = {self.biases}, scales = {self.scales}")

        logger.info(f"assim_period = {self.assim_period}")
        logger.info(f"n_snapshots = {self.n_snapshots}")
        logger.info(f"obs noise std = {self.obs_noise_std}")
        logger.info(f"missing_value = {self.missing_value}")
        logger.info(f"initial_discarded_period = {initial_discarded_period}")
        logger.info(f"max sequence length = {max_sequence_length}")
        logger.info(f"seed = {seed}")
        logger.info(
            f"clamp: use = {self.use_clipping}, min = {self.clamp_min}, max = {self.clamp_max}"
        )
        logger.info(f"use obs = {self.use_obs}")
        logger.info(f"is_output_only_last = {self.is_output_only_last}")
        logger.info(f"Input sampling interval = {self.input_sampling_interval}")

        indices = np.arange(max_sequence_length + 1) - n_snapshots
        indices = indices[indices >= initial_discarded_period]

        if is_always_start_observed:
            indices = indices[indices % assimilation_period == 0]

        self.permitted_start_indices = indices
        logger.info(f"Permitted start indices = {self.permitted_start_indices}")

    def _read_data(
        self, dir_path: str
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # result_01234 --> 01234
        i_sim = os.path.basename(dir_path).split("_")[-1]

        hr_omega = torch.from_numpy(np.load(f"{dir_path}/hr_omegas_{i_sim}.npy")).to(
            self.dtype
        )
        hr_obsrv = torch.from_numpy(np.load(f"{dir_path}/hr_obsrvs_{i_sim}.npy")).to(
            self.dtype
        )

        lr_assim = torch.from_numpy(
            np.load(
                f"{dir_path}/lr_trains_assim_period{self.assim_period:02}_{i_sim}.npy"
            )
        ).to(self.dtype)

        lr_frcst = torch.from_numpy(
            np.load(
                f"{dir_path}/lr_trains_frcst_period{self.assim_period:02}_{i_sim}.npy"
            )
        ).to(self.dtype)

        return (hr_omega, hr_obsrv, lr_assim, lr_frcst)

    def __len__(self):
        return len(self.data_dirs)

    def _preprocess(
        self, data: torch.Tensor, use_clipping: bool = False
    ) -> torch.Tensor:

        # Add channel dim, drop the last index along y, and standardize
        ret = (data[:, None, :, :-1] - self.biases) / self.scales

        if use_clipping:
            ret = torch.clamp(ret, min=self.clamp_min, max=self.clamp_max)

        # time, channel, x, y --> time, channel, y, x dims
        return ret.permute(0, 1, 3, 2)

    def __getitem__(
        self, idx: int
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        dir_path = self.data_dirs[idx]
        hr_omega, hr_obsrv, lr_assim, lr_frcst = self._read_data(dir_path)

        i_start = self.np_rng.choice(self.permitted_start_indices)

        gt, obs, lr = extract_ground_truth_and_input(
            hr_omega=hr_omega,
            hr_obsrv=hr_obsrv,
            lr_assim=lr_assim,
            lr_frcst=lr_frcst,
            i_start=i_start,
            n_snaps=self.n_snapshots,
            assim_period=self.assim_period,
        )

        if self.obs_noise_std != 0.0:
            noise = self.np_rng.normal(loc=0, scale=self.obs_noise_std, size=obs.shape)
            obs = obs + torch.from_numpy(noise).to(self.dtype)

        lr = self._preprocess(lr, use_clipping=True)
        obs = self._preprocess(obs, use_clipping=True)
        gt = self._preprocess(gt, use_clipping=self.use_clipping)

        if self.use_obs:
            obs = torch.nan_to_num(obs, nan=self.missing_value)
        else:
            obs = torch.full_like(obs, fill_value=self.missing_value)

        if self.is_output_only_last:
            return lr[-1], obs[-1], gt[-1]

        if self.input_sampling_interval is not None:
            return lr[:: self.input_sampling_interval], obs, gt

        return lr, obs, gt


class HrObservationMatrixGenerator:
    def __init__(self):
        self.obs_matrix = None

    def generate_obs_matrix(
        self,
        *,
        nx: int,
        ny: int,
        obs_init_index: int,
        obs_interval: int = 34,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        assert 0 <= obs_init_index <= obs_interval - 1

        obs_indices = np.zeros(nx * ny)
        obs_indices[obs_init_index::obs_interval] = 1.0
        obs_indices = np.where(obs_indices == 1.0)[0]

        num_obs = len(obs_indices)

        obs_matrix = torch.zeros(num_obs, nx * ny, dtype=dtype)

        for i, j in enumerate(obs_indices):
            obs_matrix[i, j] = 1.0

        p = 100 * torch.sum(obs_matrix).item() / (nx * ny)
        logger.debug(f"observatio prob = {p} [%]")

        self.obs_matrix = obs_matrix.to(device)

        return self.obs_matrix

    def generate_projection_matrix(self):

        if self.obs_matrix is None:
            raise Exception()

        return self.obs_matrix.t().mm(self.obs_matrix)


class DatasetVorticitySnapshot(Dataset):
    def __init__(
        self,
        *,
        data_dirs: typing.List[str],
        assimilation_period: int,
        obs_noise_std: float,
        obs_interval: int = 27,
        bias: float = -12.5,
        scale: float = 25.0,
        missing_value: float = 0.0,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        initial_discarded_period: int = 16,
        max_sequence_length: int = 97,
        nx: int = 128,
        ny: int = 64,
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        self.data_dirs = copy.deepcopy(data_dirs)
        self.assim_period = assimilation_period

        self.bias = bias
        self.scale = scale
        self.missing_value = missing_value
        self.missing_values = None
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        self.obs_noise_std = obs_noise_std
        self.obs_interval = obs_interval

        self.nx = nx
        self.ny = ny

        self.rnd = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.obs_matrix_generator = HrObservationMatrixGenerator()
        self.dtype = dtype

        indices = np.arange(max_sequence_length)
        indices = indices[indices >= initial_discarded_period]
        indices = indices[indices % assimilation_period == 0]

        self.permitted_indices = indices

        self.missing_values = torch.full(
            (1, self.nx, self.ny), fill_value=self.missing_value
        )

        self.is_obses = []
        for init_idx in tqdm(range(self.obs_interval)):
            self.obs_matrix_generator.generate_obs_matrix(
                nx=self.nx,
                ny=self.ny,
                obs_interval=self.obs_interval,
                obs_init_index=init_idx,
                dtype=self.dtype,
            )
            proj = self.obs_matrix_generator.generate_projection_matrix()
            ones = torch.ones(proj.shape[0], dtype=self.dtype)

            is_obs = proj.mm(ones[:, None]).reshape(self.nx, self.ny)[None, ...]
            self.is_obses.append(is_obs)

    def __len__(self):
        return len(self.data_dirs)

    def _read_data(self, dir_path: str):

        # result_01234 --> 01234
        i_sim = os.path.basename(dir_path).split("_")[-1]

        hr_omega = torch.from_numpy(np.load(f"{dir_path}/hr_omegas_{i_sim}.npy")).to(
            self.dtype
        )

        lr_frcst = torch.from_numpy(
            np.load(
                f"{dir_path}/lr_trains_frcst_period{self.assim_period:02}_{i_sim}.npy"
            )
        ).to(self.dtype)

        return hr_omega, lr_frcst

    def _preprocess(self, data: torch.Tensor) -> torch.Tensor:

        # Drop the last index along y, and then standardize
        ret = (data[..., :-1] - self.bias) / self.scale

        ret = torch.clamp(ret, min=self.clamp_min, max=self.clamp_max)

        return ret

    def __getitem__(self, idx: int):
        dir_path = self.data_dirs[idx]
        hr_omega, lr_frcst = self._read_data(dir_path)

        i = self.np_rng.choice(self.permitted_indices)
        gt = hr_omega[:, i]
        x = lr_frcst[:, i]

        gt = self._preprocess(gt)
        x = self._preprocess(x)

        assert gt.shape == (1, self.nx, self.ny)

        noise = self.np_rng.normal(loc=0, scale=self.obs_noise_std, size=gt.shape)
        noise = torch.from_numpy(noise).to(self.dtype)

        i = self.rnd.randint(0, self.obs_interval - 1)
        is_obs = self.is_obses[i]

        y = torch.where(is_obs == 1, gt + noise, self.missing_values)

        return x, y, is_obs, gt


class HrObservationMatrixGeneratorRegularInterval:
    def __init__(self):
        self.obs_matrix = None

    def generate_obs_matrix(
        self,
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

        obs_indices = np.zeros(shape=(nx, ny))
        obs_indices[init_index_x::interval_x, init_index_y::interval_y] = 1.0
        obs_indices = obs_indices.reshape(-1)
        obs_indices = np.where(obs_indices == 1.0)[0]

        num_obs = len(obs_indices)

        obs_matrix = torch.zeros(num_obs, nx * ny, dtype=dtype)

        for i, j in enumerate(obs_indices):
            obs_matrix[i, j] = 1.0

        p = 100 * torch.sum(obs_matrix).item() / (nx * ny)
        logger.debug(f"observatio prob = {p} [%]")

        self.obs_matrix = obs_matrix.to(device)

        return self.obs_matrix

    def generate_projection_matrix(self):

        if self.obs_matrix is None:
            raise Exception()

        proj = self.obs_matrix.t().mm(self.obs_matrix)

        self.obs_matrix = None

        return proj


class DatasetMakingObsInside(Dataset):
    def __init__(
        self,
        *,
        data_dirs: typing.List[str],
        assimilation_period: int,
        lr_input_sampling_interval: int,
        hr_output_sampling_interval: int,
        observation_interval: int,
        observation_noise_percentage: float,
        vorticity_bias: float,
        vorticity_scale: float,
        use_observation: bool,
        nx: int = 128,
        ny: int = 65,
        missing_value: float = 0.0,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        use_ground_truth_clamping: bool = False,
        max_sequence_length: int = 97,
        initial_discarded_period: int = 16,
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        assert assimilation_period > 0
        assert initial_discarded_period >= 0

        self.data_dirs = copy.deepcopy(data_dirs)
        self.assim_period = assimilation_period
        self.lr_input_sampling_interval = lr_input_sampling_interval
        self.hr_output_sampling_interval = hr_output_sampling_interval
        self.n_snapshots = assimilation_period + 1
        self.obs_interval = observation_interval
        self.obs_noise_std = vorticity_scale * observation_noise_percentage / 100.0
        self.vorticity_bias = vorticity_bias
        self.vorticity_scale = vorticity_scale
        self.use_obs = use_observation
        self.nx = nx
        self.ny = ny

        self.missing_value = missing_value
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.use_gt_clamp = use_ground_truth_clamping
        self.dtype = dtype

        logger.info(f"assimilation period = {self.assim_period}")
        logger.info(f"lr input sampling interval = {self.lr_input_sampling_interval}")
        logger.info(f"hr output sampling interval = {self.hr_output_sampling_interval}")
        logger.info(f"n_snapshots = {self.n_snapshots}")
        logger.info(f"observation iterval = {self.obs_interval}")
        logger.info(f"observation noise std = {self.obs_noise_std}")
        logger.info(f"use observation = {self.use_obs}")
        logger.info(f"nx = {self.nx}, ny = {self.ny}")

        logger.info(f"missing value = {self.missing_value}")
        logger.info(
            f"vorticity bias = {self.vorticity_bias}, scale = {self.vorticity_scale}"
        )
        logger.info(
            f"Clamping info: min = {self.clamp_min}, max = {self.clamp_max}, apply to ground truth = {self.use_gt_clamp}"
        )

        logger.info(f"max sequence length = {max_sequence_length}")
        logger.info(f"initial_discarded_period = {initial_discarded_period}")
        logger.info(f"Random seed = {seed}")

        self.np_rng = np.random.default_rng(seed)
        self.rnd = random.Random(seed)

        indices = np.arange(max_sequence_length + 1) - self.n_snapshots
        indices = indices[indices >= initial_discarded_period]
        indices = indices[indices % assimilation_period == 0]
        self.permitted_start_indices = indices

        logger.info(f"Permitted start indices = {self.permitted_start_indices}")

        self.obs_mat_gen = HrObservationMatrixGeneratorRegularInterval()
        self.is_obses = []
        self.obs_matrices = []

        ratio_mean = []
        for init_x in tqdm(range(self.obs_interval)):
            for init_y in range(self.obs_interval):
                mat = self.obs_mat_gen.generate_obs_matrix(
                    nx=self.nx,
                    ny=self.ny,
                    init_index_x=init_x,
                    init_index_y=init_y,
                    interval_x=self.obs_interval,
                    interval_y=self.obs_interval,
                    dtype=self.dtype,
                )
                self.obs_matrices.append(mat.clone())
                proj = self.obs_mat_gen.generate_projection_matrix()
                ones = torch.ones(proj.shape[0], dtype=self.dtype)

                is_obs = proj.mm(ones[:, None]).reshape(self.nx, self.ny)[None, ...]
                is_obs = torch.broadcast_to(
                    is_obs, size=(self.n_snapshots, self.nx, self.ny)
                )
                self.is_obses.append(is_obs)
                ratio_mean.append(torch.mean(is_obs).item())
        ratio_mean = sum(ratio_mean) / len(ratio_mean)
        logger.warning(
            f"Observation interval = {self.obs_interval}, Observation grid ratio = {ratio_mean}"
        )

    def __len__(self):
        return len(self.data_dirs)

    def _read_data(
        self, dir_path: str
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # result_01234 --> 01234
        i_sim = os.path.basename(dir_path).split("_")[-1]

        hr_omega = torch.from_numpy(np.load(f"{dir_path}/hr_omegas_{i_sim}.npy")).to(
            self.dtype
        )

        lr_assim = torch.from_numpy(
            np.load(
                f"{dir_path}/lr_trains_assim_period{self.assim_period:02}_{i_sim}.npy"
            )
        ).to(self.dtype)

        lr_frcst = torch.from_numpy(
            np.load(
                f"{dir_path}/lr_trains_frcst_period{self.assim_period:02}_{i_sim}.npy"
            )
        ).to(self.dtype)

        return (hr_omega, lr_assim, lr_frcst)

    def _extract_lr_input_and_hr_ground_truth(
        self,
        *,
        hr_omega: torch.Tensor,
        lr_assim: torch.Tensor,
        lr_frcst: torch.Tensor,
        i_start: int,
    ):
        # Check ensemble and time dimensions
        assert hr_omega.shape[:2] == lr_assim.shape[:2] == lr_frcst.shape[:2]
        assert hr_omega.shape[0] == 1  # ensemble dimension

        assert (
            i_start % self.assim_period == 0
        ), "i_start is NOT at the assimilation step"

        i_end = i_start + self.n_snapshots
        gt = hr_omega[0, i_start:i_end]  # ground truth

        # Last low-resolution input is from forecast data, which is before assimilation
        lr = lr_assim[0, i_start:i_end]
        lr[-1] = lr_frcst[0, i_end - 1]

        return lr, gt

    def _extract_observation_without_noise(self, hr_omega: torch.Tensor):

        i = self.rnd.randint(0, len(self.is_obses) - 1)
        is_obs = self.is_obses[i]
        assert is_obs.shape == hr_omega.shape
        logger.debug(f"index of is_obs = {i}")

        obs = torch.full_like(hr_omega, torch.nan)
        _tmp = torch.where(is_obs > 0, hr_omega, obs)
        obs[:: self.assim_period] = _tmp[:: self.assim_period]

        return obs

    def _preprocess(
        self, data: torch.Tensor, use_clipping: bool = False
    ) -> torch.Tensor:

        # Add channel dim, drop the last index along y, and standardize
        ret = (data[:, None, :, :-1] - self.vorticity_bias) / self.vorticity_scale

        if use_clipping:
            ret = torch.clamp(ret, min=self.clamp_min, max=self.clamp_max)

        # time, channel, x, y --> time, channel, y, x dims
        return ret.permute(0, 1, 3, 2)

    def __getitem__(self, idx: int):

        dir_path = self.data_dirs[idx]
        hr_omega, lr_assim, lr_frcst = self._read_data(dir_path)

        i_start = self.np_rng.choice(self.permitted_start_indices)

        lr, gt = self._extract_lr_input_and_hr_ground_truth(
            hr_omega=hr_omega, lr_assim=lr_assim, lr_frcst=lr_frcst, i_start=i_start
        )

        obs = self._extract_observation_without_noise(gt)
        if self.obs_noise_std > 0:
            noise = self.np_rng.normal(loc=0, scale=self.obs_noise_std, size=obs.shape)
            obs = obs + torch.from_numpy(noise).to(self.dtype)

        lr = self._preprocess(lr, use_clipping=True)
        obs = self._preprocess(obs, use_clipping=True)
        gt = self._preprocess(gt, use_clipping=self.use_gt_clamp)

        if self.use_obs:
            obs = torch.nan_to_num(obs, nan=self.missing_value)
        else:
            obs = torch.full_like(obs, fill_value=self.missing_value)

        return (
            lr[:: self.lr_input_sampling_interval],
            obs[:: self.hr_output_sampling_interval],
            gt[:: self.hr_output_sampling_interval],
        )


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

    p = 100 * torch.sum(is_obs).item() / (nx * ny)
    logger.debug(f"observatio prob = {p} [%]")

    return is_obs.to(device), obs_matrix.to(device)


class DatasetMakingObsInsideTimeseriesSplitted(Dataset):
    def __init__(
        self,
        *,
        data_dirs: typing.List[str],
        lr_kind_names: typing.List[str],
        lr_time_interval: int,
        obs_time_interval: int,
        obs_grid_interval: int,
        obs_noise_std: float,
        use_observation: bool,
        vorticity_bias: float,
        vorticity_scale: float,
        use_ground_truth_clamping: bool,
        missing_value: float = 0.0,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        nx: int = 128,
        ny: int = 65,
        max_ensemble: int = 20,
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        if obs_grid_interval <= 0:
            assert not use_observation
        else:
            assert use_observation

        self.dtype = dtype
        self.lr_time_interval = lr_time_interval
        self.obs_time_interval = obs_time_interval
        self.obs_grid_interval = obs_grid_interval if obs_grid_interval > 0 else 8
        self.obs_noise_std = obs_noise_std
        self.use_obs = use_observation
        self.vorticity_bias = vorticity_bias
        self.vorticity_scale = vorticity_scale
        self.missing_value = missing_value
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.use_gt_clamp = use_ground_truth_clamping

        logger.info(f"LR time interval = {self.lr_time_interval}")
        if obs_grid_interval > 0:
            logger.info("Obs grid interval is not replaced with 8.")
        logger.info(f"Obs time interval = {self.obs_time_interval}")
        logger.info(f"Obs grid interval = {self.obs_grid_interval}")
        logger.info(f"Obs noise std = {self.obs_noise_std}")
        logger.info(f"Use observation = {self.use_obs}")
        logger.info(f"Bias = {self.vorticity_bias}, Scale = {self.vorticity_scale}")
        logger.info(f"Use clamp for ground truth = {self.use_gt_clamp}")
        logger.info(f"Clamp: min = {self.clamp_min}, max = {self.clamp_max}")
        logger.info(f"missing value = {self.missing_value}")

        self._set_hr_file_paths(data_dirs)

        self.lr_kind_names = copy.deepcopy(lr_kind_names)
        _lst = "\n  ".join(self.lr_kind_names)
        logger.info(f"lr_kind_names = {_lst}")

        self.np_rng = np.random.default_rng(seed)
        self.rnd = random.Random(seed)

        self.max_ensemble = max_ensemble
        logger.info(f"Max ensemble = {self.max_ensemble}")

        self.is_obses = []
        self.obs_matrices = []
        ratio_mean = []

        for init_x in tqdm(range(self.obs_grid_interval)):
            for init_y in range(self.obs_grid_interval):
                is_obs, obs_mat = generate_is_obs_and_obs_matrix(
                    nx=nx,
                    ny=ny,
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
        logger.warning(
            f"Observation interval = {self.obs_grid_interval}, Observation grid ratio = {ratio_mean}"
        )

    def _set_hr_file_paths(self, data_dirs: typing.List[str]):
        lst_hr_file_paths = [
            glob.glob(f"{dir_path}/*_hr_omega.npy") for dir_path in data_dirs
        ]
        hr_file_paths = functools.reduce(lambda l1, l2: l1 + l2, lst_hr_file_paths, [])

        extracted_paths = []
        for path in hr_file_paths:
            if "start12" in path:
                continue
            if "start08" in path:
                continue
            if "start04" in path:
                continue
            if "start00" in path:
                continue
            extracted_paths.append(path)

        self.hr_file_paths = extracted_paths

    def __len__(self) -> int:
        return len(self.hr_file_paths)

    def _load_np_data(self, path_idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        hr_path = self.hr_file_paths[path_idx]
        ens_idx = self.rnd.randint(0, self.max_ensemble - 1)

        hr = np.load(hr_path)
        assert hr.shape[0] == self.max_ensemble
        hr = hr[ens_idx]
        hr = torch.from_numpy(hr).to(self.dtype)

        lr_kind = self.np_rng.choice(self.lr_kind_names, size=1, replace=False)[0]
        lr_path = hr_path.replace("hr_omega", lr_kind)
        lr = np.load(lr_path)
        assert lr.shape[0] == self.max_ensemble
        lr = lr[ens_idx]
        lr = torch.from_numpy(lr).to(self.dtype)

        logger.debug(f"Selected lr kind = {lr_kind}")
        logger.debug(f"Selected ensemble index = {ens_idx}")
        logger.debug(f"hr = {hr_path}\nlr={lr_path}")

        return lr, hr

    def _extract_observation_without_noise(self, hr_omega: torch.Tensor):

        i = self.rnd.randint(0, len(self.is_obses) - 1)
        is_obs = self.is_obses[i]
        assert is_obs.shape == hr_omega.shape[1:]

        is_obs = torch.broadcast_to(is_obs, hr_omega.shape)
        logger.debug(f"index of is_obs = {i}")

        obs = torch.full_like(hr_omega, torch.nan)
        _tmp = torch.where(is_obs > 0, hr_omega, obs)
        obs[:: self.obs_time_interval] = _tmp[:: self.obs_time_interval]

        return obs

    def _preprocess(
        self, data: torch.Tensor, use_clipping: bool = False
    ) -> torch.Tensor:

        # Add channel dim, drop the last index along y, and standardize
        ret = (data[:, None, :, :-1] - self.vorticity_bias) / self.vorticity_scale

        if use_clipping:
            ret = torch.clamp(ret, min=self.clamp_min, max=self.clamp_max)

        # time, channel, x, y --> time, channel, y, x dims
        return ret.permute(0, 1, 3, 2)

    def __getitem__(
        self, idx: int
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        lr, gt = self._load_np_data(idx)
        obs = self._extract_observation_without_noise(gt)

        if self.obs_noise_std > 0:
            noise = self.np_rng.normal(loc=0, scale=self.obs_noise_std, size=obs.shape)
            obs = obs + torch.from_numpy(noise).to(self.dtype)

        lr = self._preprocess(lr, use_clipping=True)
        obs = self._preprocess(obs, use_clipping=True)
        gt = self._preprocess(gt, use_clipping=self.use_gt_clamp)

        if self.use_obs:
            obs = torch.nan_to_num(obs, nan=self.missing_value)
        else:
            obs = torch.full_like(obs, fill_value=self.missing_value)

        return lr[:: self.lr_time_interval], obs, gt


class DatasetMakingObsInsideTimeseriesSplittedWithMixup(Dataset):
    def __init__(
        self,
        *,
        data_dirs: typing.List[str],
        lr_kind_names: typing.List[str],
        lr_time_interval: int,
        obs_time_interval: int,
        obs_grid_interval: int,
        obs_noise_std: float,
        use_observation: bool,
        vorticity_bias: float,
        vorticity_scale: float,
        use_ground_truth_clamping: bool,
        beta_dist_alpha: float,
        beta_dist_beta: float,
        use_mixup: bool,
        use_mixup_init_time: bool,
        use_lr_forecast: bool = True,
        missing_value: float = 0.0,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        nx: int = 128,
        ny: int = 65,
        max_ensemble: int = 20,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        if obs_grid_interval <= 0:
            assert not use_observation
        else:
            assert use_observation

        if use_mixup_init_time:
            assert use_mixup

        self.dtype = dtype
        self.lr_time_interval = lr_time_interval
        self.obs_time_interval = obs_time_interval
        self.obs_grid_interval = obs_grid_interval if obs_grid_interval > 0 else 8
        self.obs_noise_std = obs_noise_std
        self.use_obs = use_observation
        self.vorticity_bias = vorticity_bias
        self.vorticity_scale = vorticity_scale
        self.beta_dist_alpha = beta_dist_alpha
        self.beta_dist_beta = beta_dist_beta
        self.use_mixup = use_mixup
        self.use_mixup_init_time = use_mixup_init_time
        self.use_lr_forecast = use_lr_forecast
        self.missing_value = missing_value
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.use_gt_clamp = use_ground_truth_clamping

        logger.info(f"LR time interval = {self.lr_time_interval}")
        if obs_grid_interval > 0:
            logger.info("Obs grid interval is not replaced with 8.")
        logger.info(f"Obs time interval = {self.obs_time_interval}")
        logger.info(f"Obs grid interval = {self.obs_grid_interval}")
        logger.info(f"Obs noise std = {self.obs_noise_std}")
        logger.info(f"Use observation = {self.use_obs}")
        logger.info(f"Bias = {self.vorticity_bias}, Scale = {self.vorticity_scale}")
        logger.info(f"beta_dist_alpha = {self.beta_dist_alpha}")
        logger.info(f"beta_dist_beta = {self.beta_dist_beta}")
        logger.info(f"use_mixup = {self.use_mixup}")
        logger.info(f"use_mixup_init_time = {self.use_mixup_init_time}")
        logger.info(f"use_lr_forecast = {self.use_lr_forecast}")
        logger.info(f"Use clamp for ground truth = {self.use_gt_clamp}")
        logger.info(f"Clamp: min = {self.clamp_min}, max = {self.clamp_max}")
        logger.info(f"missing value = {self.missing_value}")

        self._set_hr_file_paths(data_dirs)
        self.str_seeds = list(
            set(map(lambda p: os.path.basename(p).split("_")[0], self.hr_file_paths))
        )

        self.lr_kind_names = copy.deepcopy(lr_kind_names)
        _lst = "\n  ".join(self.lr_kind_names)
        logger.info(f"lr_kind_names = {_lst}")

        self.max_ensemble = max_ensemble
        logger.info(f"Max ensemble = {self.max_ensemble}")

        self.is_obses = []
        self.obs_matrices = []
        ratio_mean = []

        for init_x in tqdm(range(self.obs_grid_interval)):
            for init_y in range(self.obs_grid_interval):
                is_obs, obs_mat = generate_is_obs_and_obs_matrix(
                    nx=nx,
                    ny=ny,
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
        logger.warning(
            f"Observation interval = {self.obs_grid_interval}, Observation grid ratio = {ratio_mean}"
        )

    def _set_hr_file_paths(self, data_dirs: typing.List[str]):
        lst_hr_file_paths = [
            glob.glob(f"{dir_path}/*_hr_omega.npy") for dir_path in data_dirs
        ]
        hr_file_paths = functools.reduce(lambda l1, l2: l1 + l2, lst_hr_file_paths, [])

        extracted_paths = []
        for path in hr_file_paths:
            if "start12" in path:
                continue
            if "start08" in path:
                continue
            if "start04" in path:
                continue
            if "start00" in path:
                continue
            extracted_paths.append(path)

        self.hr_file_paths = extracted_paths

    def __len__(self) -> int:
        return len(self.hr_file_paths)

    def _load_np_data(
        self, path_idx: int, ens_idx: int = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        hr_path = self.hr_file_paths[path_idx]
        if ens_idx is None:
            ens_idx = random.randint(0, self.max_ensemble - 1)

        hr = np.load(hr_path)
        assert hr.shape[0] == self.max_ensemble
        hr = hr[ens_idx]
        hr = torch.from_numpy(hr).to(self.dtype)

        lr_kind = np.random.choice(self.lr_kind_names, size=1, replace=False)[0]

        target_lr_path = hr_path.replace("hr_omega", lr_kind)
        target_lr = np.load(target_lr_path)
        assert target_lr.shape[0] == self.max_ensemble
        target_lr = target_lr[ens_idx]

        trg_seed = os.path.basename(target_lr_path).split("_")[0]
        src_seed = np.random.choice(self.str_seeds, size=1)[0]
        source_lr_path = target_lr_path.replace(trg_seed, src_seed)
        source_lrs = np.load(source_lr_path)
        source_lr = self._get_similar_source_lr(source_lrs, target_lr)

        target_lr = torch.from_numpy(target_lr).to(self.dtype)
        source_lr = torch.from_numpy(source_lr).to(self.dtype)

        logger.debug(f"Selected lr kind = {lr_kind}")
        logger.debug(f"Selected ensemble index = {ens_idx}")
        logger.debug(f"hr = {hr_path}")
        logger.debug(f"target_lr={target_lr_path}")
        logger.debug(f"source_lr={source_lr_path}")

        return target_lr, source_lr, hr

    def _get_similar_source_lr(self, source_lrs: np.ndarray, target_lr: np.ndarray):
        assert source_lrs.shape[0] == self.max_ensemble
        assert source_lrs.shape[1:] == target_lr.shape

        # Calc diffrence at the first time, (i.e., time index is zero)
        # For `target_lr` the batch dim is added as `None`
        sq_diffs = (target_lr[None, 0] - source_lrs[:, 0]) ** 2
        distances = np.sqrt(np.mean(sq_diffs, axis=(-2, -1)))  # mean over x and y

        idx = np.argmin(distances)

        logger.debug(f"source_lrs.shape = {source_lrs.shape}")
        logger.debug(f"target_lr.shape = {target_lr.shape}")
        logger.debug(f"sq_diffs = {sq_diffs.shape}")
        logger.debug(f"distances = {distances}")
        logger.debug(f"arg min idx = {idx}")

        return source_lrs[idx]

    def _extract_observation_without_noise(self, hr_omega: torch.Tensor):

        i = random.randint(0, len(self.is_obses) - 1)
        is_obs = self.is_obses[i]
        assert is_obs.shape == hr_omega.shape[1:]

        is_obs = torch.broadcast_to(is_obs, hr_omega.shape)
        logger.debug(f"index of is_obs = {i}")

        obs = torch.full_like(hr_omega, torch.nan)
        _tmp = torch.where(is_obs > 0, hr_omega, obs)
        obs[:: self.obs_time_interval] = _tmp[:: self.obs_time_interval]

        return obs

    def _preprocess(
        self, data: torch.Tensor, use_clipping: bool = False
    ) -> torch.Tensor:

        # Add channel dim, drop the last index along y, and standardize
        ret = (data[:, None, :, :-1] - self.vorticity_bias) / self.vorticity_scale

        if use_clipping:
            ret = torch.clamp(ret, min=self.clamp_min, max=self.clamp_max)

        # time, channel, x, y --> time, channel, y, x dims
        return ret.permute(0, 1, 3, 2)

    def get_pair_of_lr(
        self, path_idx: int, ens_idx: int = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        target_lr, source_lr, gt = self._load_np_data(path_idx, ens_idx)
        return target_lr, source_lr

    def _getitem(
        self, path_idx: int, ens_idx: int = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        target_lr, source_lr, gt = self._load_np_data(path_idx, ens_idx)
        obs = self._extract_observation_without_noise(gt)

        if self.obs_noise_std > 0:
            noise = np.random.normal(loc=0, scale=self.obs_noise_std, size=obs.shape)
            obs = obs + torch.from_numpy(noise).to(self.dtype)

        target_lr = self._preprocess(target_lr, use_clipping=True)
        source_lr = self._preprocess(source_lr, use_clipping=True)
        obs = self._preprocess(obs, use_clipping=True)
        gt = self._preprocess(gt, use_clipping=self.use_gt_clamp)

        if self.use_obs:
            obs = torch.nan_to_num(obs, nan=self.missing_value)
        else:
            obs = torch.full_like(obs, fill_value=self.missing_value)

        if self.use_mixup:
            source_prob = np.random.beta(
                a=self.beta_dist_alpha, b=self.beta_dist_beta, size=1
            )[0]
            logger.debug(f"source_prob = {source_prob}")

            if self.use_mixup_init_time:
                lr = target_lr
                lr[0] = source_prob * source_lr[0] + (1 - source_prob) * lr[0]
            else:
                lr = source_prob * source_lr + (1 - source_prob) * target_lr

        else:
            lr = target_lr

        lr = lr[:: self.lr_time_interval]
        if not self.use_lr_forecast:
            lr = torch.full_like(lr, fill_value=self.missing_value)

        return lr, obs, gt

    def __getitem__(
        self, idx: int
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._getitem(path_idx=idx)

    def get_specified_item(
        self, i_batch: int, i_cycle: int, start_time_index: int = 16
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        dict_hr_paths = OrderedDict(
            [
                (k, sorted(g))
                for k, g in itertools.groupby(
                    self.hr_file_paths,
                    lambda p: os.path.basename(os.path.dirname(p)),
                )
            ]
        )

        target_paths = None
        for i, ps in enumerate(dict_hr_paths.values()):
            # Each file contains 20 random members
            if i == i_batch // 20:
                target_paths = ps
                break

        # ensemble index in target 20 members.
        i_ensemble = i_batch - i_batch // 20 * 20

        path = None
        for p in target_paths:
            if f"end{i_cycle + start_time_index:02}" in p:
                path = p
                break

        i_path = self.hr_file_paths.index(path)

        logger.info(f"Target path = {path}")
        logger.info(f"i_path = {i_path}, i_ensemble = {i_ensemble}")

        return self._getitem(i_path, i_ensemble)


class DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling(Dataset):
    def __init__(
        self,
        *,
        data_dirs: typing.List[str],
        lr_kind_names: typing.List[str],
        lr_time_interval: int,
        obs_time_interval: int,
        obs_grid_interval: int,
        obs_noise_std: float,
        use_observation: bool,
        vorticity_bias: float,
        vorticity_scale: float,
        use_ground_truth_clamping: bool,
        beta_dist_alpha: float,
        beta_dist_beta: float,
        use_mixup: bool,
        use_mixup_init_time: bool,
        use_lr_forecast: bool = True,
        missing_value: float = 0.0,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        nx: int = 128,
        ny: int = 65,
        max_ensemble: int = 20,
        is_output_only_last: bool = False,
        is_last_obs_missing: bool = False,
        min_start_time_index: int = 12,
        max_start_time_index: int = 96,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        logger.info(
            "Dataset is DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling"
        )

        if obs_grid_interval <= 0:
            assert not use_observation
        else:
            assert use_observation

        if use_mixup_init_time:
            assert use_mixup

        self.dtype = dtype
        self.lr_time_interval = lr_time_interval
        self.obs_time_interval = obs_time_interval
        self.obs_grid_interval = obs_grid_interval if obs_grid_interval > 0 else 8
        self.obs_noise_std = obs_noise_std
        self.use_obs = use_observation
        self.vorticity_bias = vorticity_bias
        self.vorticity_scale = vorticity_scale
        self.beta_dist_alpha = beta_dist_alpha
        self.beta_dist_beta = beta_dist_beta
        self.use_mixup = use_mixup
        self.use_mixup_init_time = use_mixup_init_time
        self.use_lr_forecast = use_lr_forecast
        self.missing_value = missing_value
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.use_gt_clamp = use_ground_truth_clamping
        self.is_output_only_last = is_output_only_last
        self.is_last_obs_missing = is_last_obs_missing
        self.min_start_time_index = min_start_time_index
        self.max_start_time_index = max_start_time_index

        logger.info(f"LR time interval = {self.lr_time_interval}")
        if obs_grid_interval > 0:
            logger.info("Obs grid interval is not replaced with 8.")
        logger.info(f"Obs time interval = {self.obs_time_interval}")
        logger.info(f"Obs grid interval = {self.obs_grid_interval}")
        logger.info(f"Obs noise std = {self.obs_noise_std}")
        logger.info(f"Use observation = {self.use_obs}")
        logger.info(f"Bias = {self.vorticity_bias}, Scale = {self.vorticity_scale}")
        logger.info(f"beta_dist_alpha = {self.beta_dist_alpha}")
        logger.info(f"beta_dist_beta = {self.beta_dist_beta}")
        logger.info(f"use_mixup = {self.use_mixup}")
        logger.info(f"use_mixup_init_time = {self.use_mixup_init_time}")
        logger.info(f"use_lr_forecast = {self.use_lr_forecast}")
        logger.info(f"Use clamp for ground truth = {self.use_gt_clamp}")
        logger.info(f"Clamp: min = {self.clamp_min}, max = {self.clamp_max}")
        logger.info(f"missing value = {self.missing_value}")
        logger.info(f"is_output_only_last = {self.is_output_only_last}")
        logger.info(f"is_last_obs_missing = {self.is_last_obs_missing}")
        logger.info(f"min_start_time_index = {self.min_start_time_index}")
        logger.info(f"max_start_time_index = {self.max_start_time_index}")

        self._set_hr_file_paths(data_dirs)
        self.str_seeds = list(
            set(map(lambda p: os.path.basename(p).split("_")[0], self.hr_file_paths))
        )

        self.lr_kind_names = copy.deepcopy(lr_kind_names)
        _lst = "\n  ".join(self.lr_kind_names)
        logger.info(f"lr_kind_names = {_lst}")

        self.max_ensemble = max_ensemble
        logger.info(f"Max ensemble = {self.max_ensemble}")

        self._load_all_lr_data_at_init_time()

        self.is_obses = []
        self.obs_matrices = []
        ratio_mean = []

        for init_x in tqdm(range(self.obs_grid_interval)):
            for init_y in range(self.obs_grid_interval):
                is_obs, obs_mat = generate_is_obs_and_obs_matrix(
                    nx=nx,
                    ny=ny,
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
        logger.warning(
            f"Observation interval = {self.obs_grid_interval}, Observation grid ratio = {ratio_mean}"
        )

    def _set_hr_file_paths(self, data_dirs: typing.List[str]):
        lst_hr_file_paths = [
            glob.glob(f"{dir_path}/*_hr_omega_*.npy") for dir_path in data_dirs
        ]
        hr_file_paths = functools.reduce(lambda l1, l2: l1 + l2, lst_hr_file_paths, [])

        extracted_paths = []
        for path in hr_file_paths:
            if not path.endswith("_00.npy"):
                continue

            grps = re.match(
                r"seed(\d+)_start(\d+)_end", os.path.basename(path)
            ).groups()

            start_idx = int(grps[1])
            if start_idx >= self.max_start_time_index:
                continue

            is_contained = True
            for _idx in range(self.min_start_time_index+1):
                if f"start{_idx:02}_" in path:
                    is_contained = False
                    break
            if is_contained:
                extracted_paths.append(path)

        self.hr_file_paths = extracted_paths

    def __len__(self) -> int:
        return len(self.hr_file_paths)

    def _load_all_lr_data_at_init_time(self):
        self.dict_all_lr_data_at_init_time = {}

        for lr_kind in self.lr_kind_names:
            self.dict_all_lr_data_at_init_time[lr_kind] = {}
            for hr_path in tqdm(self.hr_file_paths):

                for ens_idx in range(self.max_ensemble):
                    lr_path = hr_path.replace("hr_omega", lr_kind).replace(
                        "_00.npy", f"_{ens_idx:02}.npy"
                    )

                    key = re.search(
                        r"start\d+_end\d+", os.path.basename(lr_path)
                    ).group()
                    if key not in self.dict_all_lr_data_at_init_time[lr_kind]:
                        self.dict_all_lr_data_at_init_time[lr_kind][key] = []

                    lr = np.load(lr_path)
                    self.dict_all_lr_data_at_init_time[lr_kind][key].append(
                        {"data": lr[0], "path": lr_path}
                    )
                    # `0` means the initial time

    def _load_np_data(
        self, path_idx: int, ens_idx: int = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        hr_path = self.hr_file_paths[path_idx]
        if ens_idx is None:
            ens_idx = random.randint(0, self.max_ensemble - 1)
            logger.debug(f"Random ens_idx = {ens_idx}")
        hr_path = hr_path.replace("_00.npy", f"_{ens_idx:02}.npy")
        hr = np.load(hr_path)

        lr_kind = np.random.choice(self.lr_kind_names, size=1, replace=False)[0]
        target_lr_path = hr_path.replace("hr_omega", lr_kind)
        target_lr = np.load(target_lr_path)

        key = re.search(r"start\d+_end\d+", os.path.basename(target_lr_path)).group()
        # Pass target_lr at the initial time, which has the index of `0`
        source_lr_path = self._get_similar_source_lr_path(lr_kind, key, target_lr[0])
        source_lr = np.load(source_lr_path)

        hr = torch.from_numpy(hr).to(self.dtype)
        target_lr = torch.from_numpy(target_lr).to(self.dtype)
        source_lr = torch.from_numpy(source_lr).to(self.dtype)

        logger.debug(f"Selected lr kind = {lr_kind}")
        logger.debug(f"hr = {hr_path}")
        logger.debug(f"target_lr={target_lr_path}")
        logger.debug(f"lr key = {key}")
        logger.debug(f"source_lr={source_lr_path}")

        return target_lr, source_lr, hr

    def _get_similar_source_lr_path(
        self, lr_kind: str, key: str, target_lr: np.ndarray
    ):
        all_lrs = self.dict_all_lr_data_at_init_time[lr_kind][key]

        min_path, min_norm = None, np.inf

        for i in list(
            set(np.random.randint(0, len(all_lrs), size=2 * self.max_ensemble))
        )[: self.max_ensemble]:
            data = all_lrs[i]["data"]
            path = all_lrs[i]["path"]
            assert data.shape == target_lr.shape

            norm = np.mean((data - target_lr) ** 2)
            if 0 < norm < min_norm:
                min_norm = norm
                min_path = path
                logger.debug(f"norm = {norm}, path = {min_path}")

        return min_path

    def _extract_observation_without_noise(self, hr_omega: torch.Tensor):

        i = random.randint(0, len(self.is_obses) - 1)
        is_obs = self.is_obses[i]
        assert is_obs.shape == hr_omega.shape[1:]

        is_obs = torch.broadcast_to(is_obs, hr_omega.shape)
        logger.debug(f"index of is_obs = {i}")

        obs = torch.full_like(hr_omega, torch.nan)
        _tmp = torch.where(is_obs > 0, hr_omega, obs)
        obs[:: self.obs_time_interval] = _tmp[:: self.obs_time_interval]

        return obs

    def _preprocess(
        self, data: torch.Tensor, use_clipping: bool = False
    ) -> torch.Tensor:

        # Add channel dim, drop the last index along y, and standardize
        ret = (data[:, None, :, :-1] - self.vorticity_bias) / self.vorticity_scale

        if use_clipping:
            ret = torch.clamp(ret, min=self.clamp_min, max=self.clamp_max)

        # time, channel, x, y --> time, channel, y, x dims
        return ret.permute(0, 1, 3, 2)

    def get_pair_of_lr(
        self, path_idx: int, ens_idx: int = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        target_lr, source_lr, _ = self._load_np_data(path_idx, ens_idx)
        return target_lr, source_lr

    def _getitem(
        self, path_idx: int, ens_idx: int = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        target_lr, source_lr, gt = self._load_np_data(path_idx, ens_idx)
        obs = self._extract_observation_without_noise(gt)

        if self.obs_noise_std > 0:
            noise = np.random.normal(loc=0, scale=self.obs_noise_std, size=obs.shape)
            obs = obs + torch.from_numpy(noise).to(self.dtype)

        target_lr = self._preprocess(target_lr, use_clipping=True)
        source_lr = self._preprocess(source_lr, use_clipping=True)
        obs = self._preprocess(obs, use_clipping=True)
        gt = self._preprocess(gt, use_clipping=self.use_gt_clamp)

        if self.use_obs:
            is_obs = ~torch.isnan(obs)
            obs = torch.nan_to_num(obs, nan=self.missing_value)
        else:
            obs = torch.full_like(obs, fill_value=self.missing_value)

        if self.use_mixup:
            source_prob = np.random.beta(
                a=self.beta_dist_alpha, b=self.beta_dist_beta, size=1
            )[0]
            logger.debug(f"similar source_prob = {source_prob}")

            if self.use_mixup_init_time:
                lr = target_lr
                lr[0] = source_prob * source_lr[0] + (1 - source_prob) * lr[0]
            else:
                lr = source_prob * source_lr + (1 - source_prob) * target_lr

        else:
            lr = target_lr

        if self.is_last_obs_missing:
            obs[-1] = self.missing_value

        lr = lr[:: self.lr_time_interval]
        if not self.use_lr_forecast:
            lr = torch.full_like(lr, fill_value=self.missing_value)

        if self.is_output_only_last:
            return lr[-1], obs[-1], is_obs[-1], gt[-1]

        return lr, obs, gt

    def __getitem__(
        self, idx: int
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._getitem(path_idx=idx)

    def get_specified_item(
        self,
        i_batch: int,
        i_cycle: int,
        start_time_index: int = 16,
        is_specified_end: bool = True,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        dict_hr_paths = OrderedDict(
            [
                (k, sorted(g))
                for k, g in itertools.groupby(
                    self.hr_file_paths,
                    lambda p: os.path.basename(os.path.dirname(p)),
                )
            ]
        )

        target_paths = None
        for i, ps in enumerate(dict_hr_paths.values()):
            # Each file contains 20 random members
            if i == i_batch // 20:
                target_paths = ps
                break

        # ensemble index in target 20 members.
        i_ensemble = i_batch - i_batch // 20 * 20

        path = None
        for p in target_paths:
            if is_specified_end and f"end{i_cycle + start_time_index:02}" in p:
                path = p
                break
            if (not is_specified_end) and f"start{i_cycle + start_time_index:02}" in p:
                path = p
                break
        i_path = self.hr_file_paths.index(path)

        logger.info(f"Target path = {path}")
        logger.info(f"i_path = {i_path}, i_ensemble = {i_ensemble}")

        return self._getitem(i_path, i_ensemble)


class DatasetUnsupervisedLearningPrototype(Dataset):
    def __init__(
        self,
        data_dirs: typing.List[str],
        obs_grid_interval: int,
        obs_noise_std: float,
        vorticity_bias: float,
        vorticity_scale: float,
        missing_value: float,
        clamp_min: float,
        clamp_max: float,
        use_gt_clamp: bool,
        use_mixup_for_da: bool,
        use_mixup_for_sr: bool,
        beta_dist_alpha: float,
        beta_dist_beta: float,
        discarded_max_time_index: int = 12,
        max_ensemble_index: int = 20,
        nx: int = 128,
        ny: int = 65,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        logger.info("DatasetUnsupervisedLearningPrototype is created.")

        self.obs_grid_interval = obs_grid_interval
        self.obs_noise_std = obs_noise_std
        self.vorticity_bias = vorticity_bias
        self.vorticity_scale = vorticity_scale
        self.missing_value = missing_value
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.use_gt_clamp = use_gt_clamp
        self.use_mixup_for_da = use_mixup_for_da
        self.use_mixup_for_sr = use_mixup_for_sr
        self.beta_dist_alpha = beta_dist_alpha
        self.beta_dist_beta = beta_dist_beta

        logger.info(f"obs_noise_std = {self.obs_noise_std}")
        logger.info(f"vorticity bias = {self.vorticity_bias}")
        logger.info(f"vorticity scale = {self.vorticity_scale}")
        logger.info(f"missing_value = {self.missing_value}")
        logger.info(f"clamp min = {self.clamp_min}, max = {self.clamp_max}")
        logger.info(f"use_gt_clamp = {self.use_gt_clamp}")
        logger.info(
            f"use mixup: sr = {self.use_mixup_for_sr}, da = {self.use_mixup_for_da}"
        )
        logger.info(f"Beta dist: a = {self.beta_dist_alpha}, b = {self.beta_dist_beta}")

        self.discarded_max_time_index = discarded_max_time_index
        self.max_ensemble_index = max_ensemble_index
        self.dtype = dtype

        self._set_hr_file_paths(data_dirs)
        self._load_all_lr_data_at_init_time()

        self.is_obses = []
        self.obs_matrices = []
        ratio_mean = []

        for init_x in tqdm(range(self.obs_grid_interval)):
            for init_y in range(self.obs_grid_interval):
                is_obs, obs_mat = generate_is_obs_and_obs_matrix(
                    nx=nx,
                    ny=ny,
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
        logger.info(
            f"Observation interval = {self.obs_grid_interval}, Observation grid ratio = {ratio_mean}"
        )

    def __len__(self) -> int:
        return len(self.hr_file_paths)

    def _set_hr_file_paths(self, data_dirs: typing.List[str]):
        lst_hr_file_paths = [
            sorted(glob.glob(f"{dir_path}/*_hr_omega_*.npy")) for dir_path in data_dirs
        ]
        hr_file_paths = functools.reduce(lambda l1, l2: l1 + l2, lst_hr_file_paths, [])

        extracted_paths = []
        for path in hr_file_paths:
            if not path.endswith("_00.npy"):
                continue

            is_added = True
            for _idx in range(self.discarded_max_time_index + 1):
                if f"start{_idx:02}" in path:
                    is_added = False
                    break

            if is_added:
                extracted_paths.append(path)

        self.hr_file_paths = extracted_paths

    def _get_lr_key(self, lr_path: str) -> str:
        return re.search(r"start\d+_end\d+", os.path.basename(lr_path)).group()

    def _load_all_lr_data_at_init_time(self):
        self.dict_all_lr_data_at_init_time = {}

        for hr_path in tqdm(self.hr_file_paths):

            for ens_idx in range(self.max_ensemble_index):
                lr_path = hr_path.replace("hr_omega", "lr_omega_no-noise").replace(
                    "_00.npy", f"_{ens_idx:02}.npy"
                )

                key = self._get_lr_key(lr_path)
                if key not in self.dict_all_lr_data_at_init_time:
                    self.dict_all_lr_data_at_init_time[key] = []

                lr = np.load(lr_path)
                self.dict_all_lr_data_at_init_time[key].append(
                    {"data": lr[0], "path": lr_path}
                )
                # `0` means the initial time

    def _load_np_data(self, path_idx: int, ens_idx: int = None):
        hr_path = self.hr_file_paths[path_idx]

        if ens_idx is None:
            ens_idx = random.randint(0, self.max_ensemble_index - 1)
            logger.debug(f"Random ens_idx = {ens_idx}")

        hr_path = hr_path.replace("_00.npy", f"_{ens_idx:02}.npy")
        hr = np.load(hr_path)

        target_lr_path = hr_path.replace("hr_omega", "lr_omega_no-noise")
        target_lr = np.load(target_lr_path)

        key = self._get_lr_key(target_lr_path)
        logger.debug(f"keys = {key}")

        source_lr_path = self._get_similar_source_lr_path(key, target_lr[0])
        source_lr = np.load(source_lr_path)

        logger.debug(f"hr_path = {hr_path}")
        logger.debug(f"target_lr_path = {target_lr_path}")
        logger.debug(f"source_lr_path = {source_lr_path}")

        return (
            torch.from_numpy(target_lr).to(self.dtype),
            torch.from_numpy(source_lr).to(self.dtype),
            torch.from_numpy(hr).to(self.dtype),
        )

    def _get_similar_source_lr_path(self, key: str, target_lr: np.ndarray):

        all_lrs = self.dict_all_lr_data_at_init_time[key]

        min_path, min_norm = None, np.inf

        for i in list(
            set(np.random.randint(0, len(all_lrs), size=3 * self.max_ensemble_index))
        )[: self.max_ensemble_index]:

            data = all_lrs[i]["data"]
            path = all_lrs[i]["path"]
            assert data.shape == target_lr.shape

            norm = np.mean((data - target_lr) ** 2)
            if 0 < norm < min_norm:
                min_norm = norm
                min_path = path
                logger.debug(f"norm = {norm}, path = {min_path}")

        return min_path

    def _preprocess(
        self, data: torch.Tensor, use_clipping: bool = False
    ) -> torch.Tensor:

        # Add channel dim
        ret = data[:, None]

        # Resize
        if ret.shape[-2:] == (128, 65):
            ret = F.interpolate(ret, size=[128, 64], mode="nearest")
        elif ret.shape[-2:] == (32, 17):
            ret = F.interpolate(ret, size=[32, 16], mode="nearest")
        else:
            raise Exception(f"shape = {ret.shape} is not supported.")

        # Linearly transform.
        ret = (ret - self.vorticity_bias) / self.vorticity_scale

        if use_clipping:
            if self.clamp_min is None and self.clamp_max is None:
                pass
            else:
                ret = torch.clamp(ret, min=self.clamp_min, max=self.clamp_max)

        # time, channel, x, y --> time, channel, y, x dims
        return ret.permute(0, 1, 3, 2)

    def _extract_observation_without_noise(self, hr_omega: torch.Tensor):

        i = random.randint(0, len(self.is_obses) - 1)
        logger.debug(f"index of is_obs = {i}")

        is_obs = self.is_obses[i]
        assert is_obs.shape == hr_omega.shape[1:]
        is_obs = torch.broadcast_to(is_obs, hr_omega.shape)

        nans = torch.full_like(hr_omega, torch.nan)

        return torch.where(is_obs > 0, hr_omega, nans)

    def _getitem(self, path_idx: int, ens_idx: int = None):
        target_lr, source_lr, gt = self._load_np_data(path_idx, ens_idx)
        obs = self._extract_observation_without_noise(gt)

        # Add noise before replacing nan with missing values
        noise = np.random.normal(loc=0, scale=self.obs_noise_std, size=obs.shape)
        obs = obs + torch.from_numpy(noise).to(self.dtype)

        target_lr = self._preprocess(target_lr, use_clipping=True)
        source_lr = self._preprocess(source_lr, use_clipping=True)
        gt = self._preprocess(gt, use_clipping=self.use_gt_clamp)
        obs = self._preprocess(obs, use_clipping=True)

        is_obs = (~torch.isnan(obs)).to(self.dtype)
        obs = torch.nan_to_num(obs, nan=self.missing_value)

        if self.use_mixup_for_da:
            w = np.random.beta(a=self.beta_dist_alpha, b=self.beta_dist_beta, size=1)[0]
            logger.debug(f"weight for similar lr = {w}")
            lr_for_da = (1 - w) * target_lr + w * source_lr
        else:
            lr_for_da = target_lr

        if self.use_mixup_for_sr:
            w = np.random.beta(a=self.beta_dist_alpha, b=self.beta_dist_beta, size=1)[0]
            logger.debug(f"weight for similar lr = {w}")
            lr_for_sr = (1 - w) * target_lr + w * source_lr
        else:
            lr_for_sr = target_lr

        # Return the last time index
        return lr_for_da[-1], lr_for_sr[-1], gt[-1], obs[-1], is_obs[-1]

    def __getitem__(self, idx: int):
        return self._getitem(path_idx=idx)
