from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


EPS = 1e-8


def batch_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample RMSE between pred and target, assuming shape (B, C, H, W)."""
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=(1, 2, 3)))


def batch_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample MAE between pred and target."""
    return torch.mean(torch.abs(pred - target), dim=(1, 2, 3))


def batch_corr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Per-sample Pearson correlation between pred and target.
    Both tensors are flattened per sample.
    """
    x = pred.view(pred.size(0), -1)
    y = target.view(target.size(0), -1)

    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean

    num = (x_centered * y_centered).sum(dim=1)
    denom = torch.sqrt(
        (x_centered ** 2).sum(dim=1) * (y_centered ** 2).sum(dim=1) + EPS
    )
    return num / denom


def batch_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
) -> torch.Tensor:
    """
    Per-sample PSNR. data_range is the dynamic range of the data.
    For vorticity normalized to [-1, 1], data_range=2.0.
    """
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    psnr = 20.0 * torch.log10(
        torch.tensor(data_range, device=pred.device, dtype=pred.dtype)
    ) - 10.0 * torch.log10(mse + EPS)
    return psnr


def batch_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
) -> torch.Tensor:
    """
    Per-sample global SSIM between pred and target.
    This is a simplified implementation that computes SSIM over the whole field.
    """
    x = pred
    y = target

    mu_x = x.mean(dim=(1, 2, 3))
    mu_y = y.mean(dim=(1, 2, 3))

    x_minus_mu_x = x - mu_x.view(-1, 1, 1, 1)
    y_minus_mu_y = y - mu_y.view(-1, 1, 1, 1)

    sigma_x2 = (x_minus_mu_x ** 2).mean(dim=(1, 2, 3))
    sigma_y2 = (y_minus_mu_y ** 2).mean(dim=(1, 2, 3))
    sigma_xy = (x_minus_mu_x * y_minus_mu_y).mean(dim=(1, 2, 3))

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return numerator / (denominator + EPS)


def laplacian2d(field: torch.Tensor, periodic: bool = True) -> torch.Tensor:
    """
    2D Laplacian with a 3x3 stencil.
    field: (B, C, H, W)
    """
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=field.dtype,
        device=field.device,
    ).view(1, 1, 3, 3)

    if periodic:
        padded = F.pad(field, (1, 1, 1, 1), mode="circular")
    else:
        padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(padded, kernel)


def gradients2d(
    field: torch.Tensor,
    periodic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Centered finite-difference gradients along x (height) and y (width).
    field: (B, C, H, W)
    Returns (dx, dy) with the same shape as field.
    """
    kx = torch.tensor(
        [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]],
        dtype=field.dtype,
        device=field.device,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
        dtype=field.dtype,
        device=field.device,
    ).view(1, 1, 3, 3)

    if periodic:
        padded = F.pad(field, (1, 1, 1, 1), mode="circular")
    else:
        padded = F.pad(field, (1, 1, 1, 1), mode="replicate")

    dx = F.conv2d(padded, kx)
    dy = F.conv2d(padded, ky)
    return dx, dy


def gradmag2d(field: torch.Tensor, periodic: bool = True) -> torch.Tensor:
    """Gradient magnitude |∇field|."""
    dx, dy = gradients2d(field, periodic=periodic)
    return torch.sqrt(dx ** 2 + dy ** 2 + EPS)


def batch_rmse_laplacian(
    pred: torch.Tensor,
    target: torch.Tensor,
    periodic: bool = True,
) -> torch.Tensor:
    """Per-sample RMSE of Laplacian(ω)."""
    lap_pred = laplacian2d(pred, periodic=periodic)
    lap_target = laplacian2d(target, periodic=periodic)
    return batch_rmse(lap_pred, lap_target)


def batch_rmse_gradmag(
    pred: torch.Tensor,
    target: torch.Tensor,
    periodic: bool = True,
) -> torch.Tensor:
    """Per-sample RMSE of |∇ω|."""
    gm_pred = gradmag2d(pred, periodic=periodic)
    gm_target = gradmag2d(target, periodic=periodic)
    return batch_rmse(gm_pred, gm_target)


def batch_physics_scalars(field: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute scalar physics diagnostics for each sample in a batch.
    field: (B, 1, H, W) – normalized vorticity.
    Returns (B,) tensors for each key.
    """
    B = field.size(0)
    flat = field.view(B, -1)

    mean = flat.mean(dim=1)
    var = flat.var(dim=1, unbiased=False)

    enstrophy = 0.5 * (flat ** 2).mean(dim=1)

    dx, dy = gradients2d(field)
    grad2 = dx ** 2 + dy ** 2
    grad2_flat = grad2.view(B, -1)
    palinstrophy = 0.5 * grad2_flat.mean(dim=1)

    std = torch.sqrt(var + EPS)
    centered = flat - mean.unsqueeze(1)
    m3 = (centered ** 3).mean(dim=1)
    m4 = (centered ** 4).mean(dim=1)

    skewness = m3 / (std ** 3 + EPS)
    kurtosis = m4 / (std ** 4 + EPS)

    return {
        "mean": mean,
        "var": var,
        "enstrophy": enstrophy,
        "palinstrophy": palinstrophy,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def batch_physics_errors(
    sr: torch.Tensor,
    hr: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Per-sample absolute and relative errors of physics scalars between SR and HR.
    Returns keys like 'enstrophy_abs_err', 'enstrophy_rel_err', etc.
    """
    sr_scalars = batch_physics_scalars(sr)
    hr_scalars = batch_physics_scalars(hr)

    out: Dict[str, torch.Tensor] = {}
    for name, sr_vals in sr_scalars.items():
        hr_vals = hr_scalars[name]
        abs_err = torch.abs(sr_vals - hr_vals)
        rel_err = abs_err / (torch.abs(hr_vals) + EPS)
        out[f"{name}_abs_err"] = abs_err
        out[f"{name}_rel_err"] = rel_err
    return out


def summarize_metric_batches(metric_lists: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Given dict of metric_name -> list of per-sample values,
    return metric_name_mean and metric_name_std entries.
    """
    summary: Dict[str, float] = {}
    for name, values in metric_lists.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            summary[f"{name}_mean"] = float("nan")
            summary[f"{name}_std"] = float("nan")
        else:
            summary[f"{name}_mean"] = float(arr.mean())
            summary[f"{name}_std"] = float(arr.std())
    return summary


def evaluate_rmse_and_time(
    model,
    dataloader,
    limit_batches: Optional[int] = None,
) -> Dict[str, float]:
    rmse_values = []
    n_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            visuals = model.infer(batch)
            sr = visuals["SR"]
            hr = visuals["HR"]
            rmse_batch = batch_rmse(sr, hr).cpu().numpy()
            rmse_values.extend(rmse_batch.tolist())
            n_samples += len(rmse_batch)

            if limit_batches is not None and (i + 1) >= limit_batches:
                break

    elapsed = time.time() - start_time
    rmse_mean = float(np.mean(rmse_values)) if rmse_values else float("nan")
    time_per_sample = elapsed / max(n_samples, 1)

    return {
        "rmse": rmse_mean,
        "samples": n_samples,
        "elapsed_sec": elapsed,
        "time_per_sample_sec": time_per_sample,
    }


def save_metrics(metrics: Dict[str, float], output_csv: Path):
    """
    Save metrics to CSV in a tall (metric, value) format.

    The resulting file has two columns:
        metric,value
        name1,val1
        name2,val2
        ...
    This is easier to read in plain text than a single wide row.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    names = list(metrics.keys())
    values = [metrics[name] for name in names]
    df = pd.DataFrame({"metric": names, "value": values})
    df.to_csv(output_csv, index=False)
