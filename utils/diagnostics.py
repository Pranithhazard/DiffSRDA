from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from utils.metrics import laplacian2d


def compute_laplacian_field(
    omega: torch.Tensor,
    periodic: bool = True,
) -> torch.Tensor:
    """
    Compute vorticity diffusion field ∇²ω.

    Parameters
    ----------
    omega : torch.Tensor
        Vorticity field of shape (B, 1, H, W).
    periodic : bool, optional
        Whether to assume periodic boundaries, by default True.

    Returns
    -------
    torch.Tensor
        Laplacian field with the same shape as omega.
    """
    return laplacian2d(omega, periodic=periodic)


def compute_enstrophy_field(omega: torch.Tensor) -> torch.Tensor:
    """
    Compute pointwise enstrophy field 0.5 * omega^2.

    Parameters
    ----------
    omega : torch.Tensor
        Vorticity field of shape (B, 1, H, W).

    Returns
    -------
    torch.Tensor
        Enstrophy field of shape (B, 1, H, W).
    """
    return 0.5 * omega ** 2


def compute_error_fields(
    sr: torch.Tensor,
    hr: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute error fields between super-resolved (SR) and high-resolution (HR) vorticity.

    Parameters
    ----------
    sr : torch.Tensor
        SR vorticity field of shape (B, 1, H, W).
    hr : torch.Tensor
        HR vorticity field of shape (B, 1, H, W).

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with keys:
            'err'     : sr - hr
            'abs_err' : |sr - hr|
            'sq_err'  : (sr - hr)^2
    """
    err = sr - hr
    abs_err = torch.abs(err)
    sq_err = err ** 2
    return {
        "err": err,
        "abs_err": abs_err,
        "sq_err": sq_err,
    }


def compute_laplacian_rmse_time_series(
    pred: torch.Tensor,
    truth: torch.Tensor,
    *,
    periodic: bool = True,
    ensemble_reduce: str = "mean",
) -> torch.Tensor:
    """
    Compute per-time-step RMSE of Laplacian(ω) between a prediction and truth.

    This is a convenience helper for plotting "RMSE of Laplacian of vorticity vs time".

    Supported shapes
    ----------------
    pred:
      - (T, H, W)               deterministic time series
      - (B, T, H, W)            ensemble time series (e.g., diffusion samples)
      - (T, 1, H, W) or (B, T, 1, H, W)
    truth:
      - (T, H, W) or (T, 1, H, W)

    The Laplacian operator is the same 3x3 stencil used in `utils.metrics.laplacian2d`
    (unit grid spacing). For comparisons across methods on the same grid, the constant
    scaling does not affect relative ranking.

    Parameters
    ----------
    pred : torch.Tensor
        Prediction tensor (see supported shapes above).
    truth : torch.Tensor
        Truth tensor (see supported shapes above).
    periodic : bool, optional
        Whether to use periodic boundaries (circular padding), by default True.
    ensemble_reduce : str, optional
        How to reduce an ensemble dimension if present in pred.
        One of: {"mean", "median"}, by default "mean".

    Returns
    -------
    torch.Tensor
        A 1D tensor of length T containing RMSE(∇²ω) at each time step.
    """

    def _squeeze_channel(x: torch.Tensor) -> torch.Tensor:
        if x.ndim >= 4 and x.size(-3) == 1:
            return x.squeeze(-3)
        return x

    pred = _squeeze_channel(pred)
    truth = _squeeze_channel(truth)

    # Reduce ensemble dimension if present.
    if pred.ndim == 4:
        # (B, T, H, W)
        if ensemble_reduce == "mean":
            pred = pred.mean(dim=0)
        elif ensemble_reduce == "median":
            pred = pred.median(dim=0).values
        else:
            raise ValueError(f"Unsupported ensemble_reduce={ensemble_reduce!r}")
    elif pred.ndim == 5:
        # (B, T, 1, H, W) -> squeeze channel then reduce
        pred = pred.squeeze(2)
        if ensemble_reduce == "mean":
            pred = pred.mean(dim=0)
        elif ensemble_reduce == "median":
            pred = pred.median(dim=0).values
        else:
            raise ValueError(f"Unsupported ensemble_reduce={ensemble_reduce!r}")

    if pred.ndim != 3:
        raise ValueError(f"pred must be (T,H,W) after reduction, got shape {tuple(pred.shape)}")

    if truth.ndim == 4:
        # (T, 1, H, W) -> (T, H, W)
        truth = truth.squeeze(1)
    if truth.ndim != 3:
        raise ValueError(f"truth must be (T,H,W), got shape {tuple(truth.shape)}")

    if pred.shape != truth.shape:
        raise ValueError(f"pred and truth must have the same shape, got {tuple(pred.shape)} vs {tuple(truth.shape)}")

    # Treat time as batch: (T, 1, H, W)
    pred_t = pred.unsqueeze(1)
    truth_t = truth.unsqueeze(1)

    lap_pred = laplacian2d(pred_t, periodic=periodic)
    lap_truth = laplacian2d(truth_t, periodic=periodic)

    rmse = torch.sqrt(torch.mean((lap_pred - lap_truth) ** 2, dim=(1, 2, 3)))
    return rmse


def _ensure_batch_3d(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array has shape (B, H, W).
    """
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")


def compute_isotropic_enstrophy_spectrum(
    omega_np: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    n_bins: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute isotropic enstrophy spectrum E(k) from vorticity field(s).

    Parameters
    ----------
    omega_np : np.ndarray
        Vorticity field in shape (H, W) or (B, H, W).
    dx, dy : float, optional
        Grid spacing in x and y directions, by default 1.0.
    n_bins : int, optional
        Number of radial wavenumber bins. If None, uses min(H, W)//2.

    Returns
    -------
    k_bins : np.ndarray
        1D array of radial wavenumber bin centers.
    E_k : np.ndarray
        1D array of isotropic enstrophy spectral density, averaged over batch if B>1.
    """
    omega_bhw = _ensure_batch_3d(omega_np.astype(np.float64))
    B, H, W = omega_bhw.shape

    if n_bins is None:
        n_bins = max(4, min(H, W) // 2)

    # Wavenumber grids
    kx = 2.0 * np.pi * np.fft.fftfreq(W, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(H, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_mag = np.sqrt(kx_grid ** 2 + ky_grid ** 2)
    k_flat = k_mag.ravel()

    # Radial bins
    k_max = k_flat.max()
    bin_edges = np.linspace(0.0, k_max, n_bins + 1)
    bin_indices = np.digitize(k_flat, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    E_accum = np.zeros(n_bins, dtype=np.float64)

    for b in range(B):
        omega = omega_bhw[b]
        # 2D FFT and power spectrum
        omega_hat = np.fft.fftn(omega)
        power = np.abs(omega_hat) ** 2

        power_flat = power.ravel()
        # Accumulate power in radial bins
        E_sum = np.bincount(bin_indices, weights=power_flat, minlength=n_bins)
        count = np.bincount(bin_indices, minlength=n_bins)
        # Avoid division by zero
        nonzero = count > 0
        E_spec = np.zeros_like(E_sum)
        E_spec[nonzero] = E_sum[nonzero] / count[nonzero]
        # Enstrophy spectrum has 0.5 factor
        E_accum += 0.5 * E_spec

    E_k = E_accum / float(B)
    # Bin centers
    k_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return k_bins, E_k


def estimate_pdf(
    data: np.ndarray,
    bins: int = 100,
    range: Tuple[float, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate PDF using a histogram.

    Parameters
    ----------
    data : np.ndarray
        1D array of samples.
    bins : int, optional
        Number of bins, by default 100.
    range : tuple, optional
        (min, max) range for the histogram. If None, uses data min/max.

    Returns
    -------
    bin_centers : np.ndarray
        Centers of the histogram bins.
    pdf : np.ndarray
        Estimated probability density values.
    """
    data = np.asarray(data).ravel()
    if range is None:
        data_min = float(data.min())
        data_max = float(data.max())
        range = (data_min, data_max)

    hist, edges = np.histogram(data, bins=bins, range=range, density=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_centers, hist


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compute discrete KL divergence KL(P || Q) for distributions on the same support.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    return float(np.sum(p * np.log(p / q)))


def wasserstein_1d(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Approximate 1D Wasserstein-1 distance between empirical distributions of x and y.

    This implementation uses quantiles on a common grid in [0, 1].
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if x.size == 0 or y.size == 0:
        return float("nan")

    # Common quantile grid
    n_q = min(x.size, y.size, 1000)
    qs = np.linspace(0.0, 1.0, n_q, endpoint=True)

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    # Quantile positions for each empirical sample
    # Since we only need the interpolated quantiles, use np.interp directly.
    x_q = np.interp(qs, np.linspace(0.0, 1.0, x_sorted.size), x_sorted)
    y_q = np.interp(qs, np.linspace(0.0, 1.0, y_sorted.size), y_sorted)

    return float(np.mean(np.abs(x_q - y_q)))
