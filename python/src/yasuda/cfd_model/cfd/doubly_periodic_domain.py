import sys
import typing

import numpy as np
import torch
from src.yasuda.cfd_model.cfd.abstract_cfd_model import AbstractCfdModel
from src.yasuda.cfd_model.fft.doubly_periodic_domain import NumpyFftCalculator, TorchFftCalculator
from src.yasuda.cfd_model.time_integration.runge_kutta import runge_kutta_2nd_order

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class NumpySpectralModel2D(AbstractCfdModel):
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        coef_diffusion: float,
        order_diffusion: int,
        norm: typing.Literal["backward", "ortho", "forward"] = "ortho",
        dtype: np.dtype = np.complex128
    ):
        self.nx = nx
        self.ny = ny
        self.coef_diffusion = coef_diffusion
        self.order_diffusion = order_diffusion

        self.__fft = NumpyFftCalculator(nx=nx, ny=ny, norm=norm, dtype=dtype)
        self.__time_integrator = runge_kutta_2nd_order
        self.__diffusion_operator = coef_diffusion * np.power(
            self.__fft.k2, order_diffusion
        )

        self.ne = 0
        self.t = None
        self.spec_omega = None
        self.omega = None
        self.u = None
        self.v = None

    @property
    def time(self):
        return self.t

    @property
    def vorticity(self):
        return self.omega

    @property
    def state_size(self):
        return self.ne, self.nx, self.ny

    def _time_derivative(self, t: float, spec_omega: np.ndarray) -> np.ndarray:
        spec_adv = self.__fft.calculate_advection_from_spec_omega(
            spec_omega, apply_fft=True
        )
        return -spec_adv - self.__diffusion_operator * spec_omega

    def calc_grid_data(self):
        self.omega = self.__fft.apply_ifft2(self.spec_omega)
        self.u, self.v = self.__fft.calculate_uv_from_omega(self.omega)

    def initialize(self, t0: float, omega0: np.ndarray):
        assert isinstance(t0, float)
        assert omega0.ndim == 3  # batch, x, y dims
        assert omega0.shape[-2] == self.nx
        assert omega0.shape[-1] == self.ny

        self.t = t0
        self.ne = omega0.shape[0]
        self.spec_omega = self.__fft.apply_fft2(omega0)

    def time_integrate(self, dt: float, nt: int, hide_progress_bar: bool = False):
        for _ in tqdm(range(nt), disable=hide_progress_bar):
            self.spec_omega = self.__time_integrator(
                dt=dt, t=self.t, x=self.spec_omega, dxdt=self._time_derivative
            )
            self.t += dt


class TorchSpectralModel2D(AbstractCfdModel):
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        coef_diffusion: float,
        order_diffusion: int,
        device: str,
        norm: typing.Literal["backward", "ortho", "forward"] = "ortho",
        dtype: torch.dtype = torch.complex128
    ):
        self.nx = nx
        self.ny = ny
        self.coef_diffusion = coef_diffusion
        self.order_diffusion = order_diffusion
        self.device = device

        self.__fft = TorchFftCalculator(
            nx=nx, ny=ny, norm=norm, dtype=dtype, device=device
        )
        self.__time_integrator = runge_kutta_2nd_order
        self.__diffusion_operator = coef_diffusion * torch.pow(
            self.__fft.k2, order_diffusion
        )

        self.ne = 0
        self.t = None
        self.spec_omega = None
        self.omega = None
        self.u = None
        self.v = None

    @property
    def time(self):
        return self.t

    @property
    def vorticity(self):
        return self.omega

    @property
    def state_size(self):
        return self.ne, self.nx, self.ny

    def _time_derivative(self, t: float, spec_omega: torch.Tensor) -> torch.Tensor:
        spec_adv = self.__fft.calculate_advection_from_spec_omega(
            spec_omega, apply_fft=True
        )
        return -spec_adv - self.__diffusion_operator * spec_omega

    def calc_grid_data(self):
        self.omega = self.__fft.apply_ifft2(self.spec_omega)
        self.u, self.v = self.__fft.calculate_uv_from_omega(self.omega)

    def initialize(self, t0: float, omega0: torch.Tensor):
        assert isinstance(t0, float)
        assert omega0.ndim == 3  # batch, x, y dims
        assert omega0.shape[-2] == self.nx
        assert omega0.shape[-1] == self.ny

        self.ne = omega0.shape[0]
        self.t = t0
        self.spec_omega = self.__fft.apply_fft2(omega0.to(self.device))

    def time_integrate(self, dt: float, nt: int, hide_progress_bar: bool = False):
        for _ in tqdm(range(nt), disable=hide_progress_bar):
            self.spec_omega = self.__time_integrator(
                dt=dt, t=self.t, x=self.spec_omega, dxdt=self._time_derivative
            )
            self.t += dt