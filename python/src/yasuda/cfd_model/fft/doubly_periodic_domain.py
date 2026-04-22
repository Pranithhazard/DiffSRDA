import typing

import numpy as np
import torch
from src.yasuda.cfd_model.fft.abstract_fft_calculator import AbstractFftCalculator


def get_wavenumber(idx: int, total_num: int) -> int:
    if idx <= total_num // 2:
        return idx
    return idx - total_num


class NumpyFftCalculator(AbstractFftCalculator):
    def __init__(
        self,
        nx: int,
        ny: int,
        norm: typing.Literal["backward", "ortho", "forward"] = "ortho",
        dtype: np.dtype = np.complex128,
    ):
        self.nx = nx
        self.ny = ny
        self.norm = norm
        self.dtype = dtype

        self.jkx = np.zeros((1, nx, ny // 2 + 1), dtype=dtype)
        self.jky = np.zeros((1, nx, ny // 2 + 1), dtype=dtype)
        self.k2 = np.zeros((1, nx, ny // 2 + 1), dtype=dtype)
        self.coef_u = np.zeros((1, nx, ny // 2 + 1), dtype=dtype)
        self.coef_v = np.zeros((1, nx, ny // 2 + 1), dtype=dtype)
        self.filter = np.ones(
            (1, nx, ny // 2 + 1), dtype=dtype
        )  # filter to remove aliasing errors

        for i in range(nx):
            kx = get_wavenumber(i, nx)
            for j in range(ny // 2 + 1):
                ky = get_wavenumber(j, ny)

                self.jkx[0, i, j] = kx * 1j
                self.jky[0, i, j] = ky * 1j

                k2 = kx**2 + ky**2
                self.k2[0, i, j] = k2

                if k2 != 0:
                    self.coef_u[0, i, j] = ky * 1j / k2
                    self.coef_v[0, i, j] = -kx * 1j / k2

                if np.abs(kx) > nx // 3 or np.abs(ky) > ny // 3:
                    self.filter[0, i, j] = 0.0

    def apply_fft2(self, data: np.ndarray) -> np.ndarray:
        assert data.ndim == 3  # batch, x, and y dims
        assert data.shape[-2] == self.nx
        assert data.shape[-1] == self.ny

        spec = np.fft.rfft2(data, axes=(-2, -1), norm=self.norm)

        return spec * self.filter

    def apply_ifft2(self, data: np.ndarray) -> np.ndarray:
        assert data.ndim == 3  # batch, x, and y dims
        assert data.shape[-2] == self.nx
        assert data.shape[-1] == (self.ny // 2 + 1)

        return np.fft.irfft2(data, axes=(-2, -1), norm=self.norm)

    def calculate_x_derivative(self, data: np.ndarray) -> np.ndarray:
        spec = self.apply_fft2(data)
        return np.fft.irfft2(spec * self.jkx, axes=(-2, -1), norm=self.norm)

    def calculate_y_derivative(self, data: np.ndarray) -> np.ndarray:
        spec = self.apply_fft2(data)
        return np.fft.irfft2(spec * self.jky, axes=(-2, -1), norm=self.norm)

    def calculate_uv_from_omega(
        self, omega: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        spec = self.apply_fft2(omega)
        u = np.fft.irfft2(spec * self.coef_u, axes=(-2, -1), norm=self.norm)
        v = np.fft.irfft2(spec * self.coef_v, axes=(-2, -1), norm=self.norm)
        return u, v

    def calculate_advection_from_spec_omega(
        self, spec: np.ndarray, apply_fft: bool = True
    ) -> np.ndarray:

        assert spec.ndim == 3  # batch, x, and y dims
        assert spec.shape[-2] == self.nx
        assert spec.shape[-1] == (self.ny // 2 + 1)

        d_omega_dx = np.fft.irfft2(spec * self.jkx, axes=(-2, -1), norm=self.norm)
        d_omega_dy = np.fft.irfft2(spec * self.jky, axes=(-2, -1), norm=self.norm)

        u = np.fft.irfft2(spec * self.coef_u, axes=(-2, -1), norm=self.norm)
        v = np.fft.irfft2(spec * self.coef_v, axes=(-2, -1), norm=self.norm)

        adv = u * d_omega_dx + v * d_omega_dy

        if not apply_fft:
            return adv

        return self.apply_fft2(adv)

    def calculate_advection_from_grid_omega(
        self, omega: np.ndarray, apply_fft: bool = True
    ) -> np.ndarray:
        spec = self.apply_fft2(omega)
        return self.calculate_advection_from_spec_omega(spec, apply_fft)


class TorchFftCalculator(AbstractFftCalculator):
    def __init__(
        self,
        nx: int,
        ny: int,
        norm: typing.Literal["backward", "ortho", "forward"] = "ortho",
        dtype: torch.dtype = torch.complex128,
        device: str = "cpu",
    ):
        self.nx = nx
        self.ny = ny

        half_ny = ny // 2 + 1
        self.half_ny = half_ny

        self.norm = norm
        self.dtype = dtype

        self.jkx = torch.zeros((1, nx, half_ny), dtype=dtype, device=device)
        self.jky = torch.zeros((1, nx, half_ny), dtype=dtype, device=device)
        self.k2 = torch.zeros((1, nx, half_ny), dtype=dtype, device=device)

        self.coef_u = torch.zeros((1, nx, half_ny), dtype=dtype, device=device)
        self.coef_v = torch.zeros((1, nx, half_ny), dtype=dtype, device=device)

        # filter to remove aliasing errors
        self.filter = torch.ones((1, nx, half_ny), dtype=dtype, device=device)

        for i in range(nx):
            kx = get_wavenumber(i, nx)
            for j in range(half_ny):
                ky = get_wavenumber(j, ny)

                self.jkx[0, i, j] = kx * 1j
                self.jky[0, i, j] = ky * 1j

                k2 = kx**2 + ky**2
                self.k2[0, i, j] = k2

                if k2 != 0:
                    self.coef_u[0, i, j] = ky * 1j / k2
                    self.coef_v[0, i, j] = -kx * 1j / k2

                if abs(kx) > nx // 3 or abs(ky) > ny // 3:
                    self.filter[0, i, j] = 0.0

    def apply_fft2(self, grid_data: torch.Tensor) -> torch.Tensor:
        assert grid_data.ndim == 3  # batch, x, and y dims
        assert grid_data.shape[-2] == self.nx
        assert grid_data.shape[-1] == self.ny

        spec = torch.fft.rfft2(grid_data, dim=(-2, -1), norm=self.norm)

        # Filter is applied to remove aliasing errors.
        return spec * self.filter

    def apply_ifft2(self, spec_data: torch.Tensor) -> torch.Tensor:
        assert spec_data.ndim == 3  # batch, x, and y dims
        assert spec_data.shape[-2] == self.nx
        assert spec_data.shape[-1] == self.half_ny

        return torch.fft.irfft2(spec_data, dim=(-2, -1), norm=self.norm)

    def calculate_x_derivative(self, grid_data: torch.Tensor) -> torch.Tensor:
        spec = self.apply_fft2(grid_data)
        return self.apply_ifft2(spec * self.jkx)

    def calculate_y_derivative(self, grid_data: torch.Tensor) -> torch.Tensor:
        spec = self.apply_fft2(grid_data)
        return self.apply_ifft2(spec * self.jky)

    def calculate_uv_from_omega(
        self, grid_omega: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        spec = self.apply_fft2(grid_omega)
        u = self.apply_ifft2(spec * self.coef_u)
        v = self.apply_ifft2(spec * self.coef_v)

        return u, v

    def calculate_advection_from_spec_omega(
        self, spec_omega: torch.Tensor, apply_fft: bool = True
    ) -> torch.Tensor:

        d_omega_dx = self.apply_ifft2(spec_omega * self.jkx)
        d_omega_dy = self.apply_ifft2(spec_omega * self.jky)

        u = self.apply_ifft2(spec_omega * self.coef_u)
        v = self.apply_ifft2(spec_omega * self.coef_v)

        adv = u * d_omega_dx + v * d_omega_dy

        if not apply_fft:
            return adv

        return self.apply_fft2(adv)

    def calculate_advection_from_grid_omega(
        self, grid_omega: torch.Tensor, apply_fft: bool = True
    ) -> torch.Tensor:
        spec = self.apply_fft2(grid_omega)
        return self.calculate_advection_from_spec_omega(spec, apply_fft)