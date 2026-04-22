import numpy as np
import torch
import torch.nn.functional as F

from src.yasuda.cfd_model.cfd.abstract_src.yasuda.cfd_model import AbstractCfdModel
from src.yasuda.cfd_model.interpolator.torch_interpolator import interpolate
from src.yasuda.cfd_model.enkf.observation_matrix import HrObservationMatrixGenerator


def _make_observation_with_noise(
    observation: torch.Tensor,
    generator: torch.Generator,
    noise_amplitude: float,
):

    obs = observation.reshape(-1)

    size = obs.size()[0]
    device = obs.device

    noise = noise_amplitude * torch.randn(size, generator=generator).to(device)

    return obs + noise


def _calc_forecast_stats(state: torch.Tensor):

    assert state.dim() >= 2

    # num of ensemble members
    ne = state.shape[0]
    assert ne > 1

    forecast_all = state.reshape(ne, -1)

    forecast_mean = torch.mean(forecast_all, dim=0, keepdim=True)
    forecast_anomaly = forecast_all - forecast_mean
    forecast_covariance = forecast_anomaly.t().mm(forecast_anomaly) / (ne - 1)

    return forecast_mean, forecast_anomaly, forecast_covariance


def _calc_obs_covariance(obs_noise_std: float, size: int, device: str):
    obs_covariance = obs_noise_std**2 * torch.eye(size)
    assert obs_covariance.shape == (size, size)
    return obs_covariance.to(device)


def _calc_kalman_gain(
    forecast_cov: torch.Tensor, obs_cov: torch.Tensor, obs_matrix: torch.Tensor
):
    assert forecast_cov.ndim == obs_cov.ndim == obs_matrix.ndim == 2

    _cov = obs_matrix.mm(forecast_cov)
    _cov = _cov.mm(obs_matrix.t())

    _inv = torch.linalg.inv(_cov + obs_cov)

    kalman_gain = _inv.mm(obs_matrix).mm(forecast_cov)

    return kalman_gain


def _assimilate(
    *,
    observation: torch.Tensor,
    model_state: torch.Tensor,
    obs_noise_std_factor: float,
    obs_matrix: torch.Tensor,
    generator: torch.Generator = None,
    inflation: float = 1.0,
):

    assert model_state.dim() >= 2
    assert obs_matrix.dim() == 2
    assert np.cumprod(observation.shape)[-1] == obs_matrix.shape[0]
    assert np.cumprod(model_state.shape[1:])[-1] == obs_matrix.shape[1]

    forecast_mean, forecast_anomaly, forecast_cov = _calc_forecast_stats(model_state)
    forecast_cov *= inflation

    max_var = torch.max(torch.diagonal(forecast_cov, dim1=0, dim2=1))
    obs_noise_std = np.sqrt(max_var.item()) * obs_noise_std_factor

    obs_size = observation.numel()
    device = forecast_cov.device
    obs_cov = _calc_obs_covariance(obs_noise_std, obs_size, device)

    kalman_gain = _calc_kalman_gain(forecast_cov, obs_cov, obs_matrix)

    obs = _make_observation_with_noise(observation, generator, obs_noise_std)
    innovation = obs - forecast_mean.mm(obs_matrix.t())
    analysis_mean = forecast_mean + (innovation).mm(kalman_gain)

    analysis_anomaly = forecast_anomaly - 0.5 * forecast_anomaly.mm(obs_matrix.t()).mm(
        kalman_gain
    )

    analysis_all = analysis_mean + analysis_anomaly

    return analysis_all, analysis_mean


def assimilate(
    *,
    hr_model: AbstractCfdModel,
    lr_ens_model: AbstractCfdModel,
    obs_matrix_generator: HrObservationMatrixGenerator,
    obs_std_factor: float,
    inflation: float,
    rand_generator: torch.Generator,
    device: str,
):
    _, hr_nx, hr_ny = hr_model.state_size
    lr_ne, lr_nx, lr_ny = lr_ens_model.state_size

    # Map lr model state to hr space
    lr_state = interpolate(lr_ens_model.omega, nx=hr_nx, ny=hr_ny).reshape(lr_ne, -1)

    obs_matrix = obs_matrix_generator.generate_obs_matrix(
        nx=hr_nx, ny=hr_ny, device=device
    )
    obs = obs_matrix.mm(hr_model.omega.reshape(-1, 1))

    analysis_all, _ = _assimilate(
        observation=obs,
        model_state=lr_state,
        obs_noise_std_factor=obs_std_factor,
        obs_matrix=obs_matrix,
        generator=rand_generator,
        inflation=inflation,
    )

    analysis_all = analysis_all.reshape(lr_ne, hr_nx, hr_ny)
    omega_all = interpolate(analysis_all, nx=lr_nx, ny=lr_ny)

    t = lr_ens_model.t
    lr_ens_model.initialize(t0=t, omega0=omega_all)
    lr_ens_model.calc_grid_data()


def assimilate_with_existing_data(
    *,
    hr_omega: torch.Tensor,
    lr_ens_model: AbstractCfdModel,
    obs_matrix: torch.Tensor,
    obs_std_factor: float,
    inflation: float,
    rand_generator: torch.Generator,
):
    assert hr_omega.ndim == 2
    assert obs_matrix.ndim == 2

    hr_nx, hr_ny = hr_omega.shape
    assert obs_matrix.shape[1] == hr_nx * hr_ny

    lr_ne, lr_nx, lr_ny = lr_ens_model.state_size

    # Map lr model state to hr space
    lr_state = interpolate(lr_ens_model.omega, nx=hr_nx, ny=hr_ny).reshape(lr_ne, -1)

    obs = obs_matrix.mm(hr_omega.reshape(-1, 1))

    analysis_all, _ = _assimilate(
        observation=obs,
        model_state=lr_state,
        obs_noise_std_factor=obs_std_factor,
        obs_matrix=obs_matrix,
        generator=rand_generator,
        inflation=inflation,
    )

    analysis_all = analysis_all.reshape(lr_ne, hr_nx, hr_ny)
    omega_all = interpolate(analysis_all, nx=lr_nx, ny=lr_ny)

    t = lr_ens_model.t
    lr_ens_model.initialize(t0=t, omega0=omega_all)
    lr_ens_model.calc_grid_data()