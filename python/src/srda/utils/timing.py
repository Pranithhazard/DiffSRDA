from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch


DeviceLike = Union[str, torch.device]


def _is_cuda(device: Optional[DeviceLike]) -> bool:
    if device is None:
        return torch.cuda.is_available()
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return str(device).startswith("cuda")


def _cuda_sync(device: Optional[DeviceLike]) -> None:
    if _is_cuda(device) and torch.cuda.is_available():
        torch.cuda.synchronize()


class CudaWallTimer:
    """
    Wall-clock timer that optionally synchronizes CUDA before/after measuring.
    This avoids under-reporting GPU work due to asynchronous launches.
    """

    def __init__(self, device: Optional[DeviceLike] = None, enabled: bool = True):
        self._device = device
        self._enabled = enabled
        self._t0: Optional[float] = None

    def start(self) -> None:
        if not self._enabled:
            return
        _cuda_sync(self._device)
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        if not self._enabled:
            return 0.0
        if self._t0 is None:
            raise RuntimeError("CudaWallTimer.stop() called before start().")
        _cuda_sync(self._device)
        t1 = time.perf_counter()
        dt = t1 - self._t0
        self._t0 = None
        return float(dt)


def _safe_mean(values: Sequence[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_std(values: Sequence[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.std(values))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


@dataclass
class CycleTimingRow:
    seed: int
    i_cycle: int
    cycle_index: int
    sampling_steps: int
    cycle_total_sec: float
    sampling_sec: float
    sampling_ms_per_step: float
    lr_forecast_sec: float = 0.0


@dataclass
class SeedTimingSummary:
    seed: int
    total_sec: float
    num_cycles: int
    warmup_cycles: int
    mean_cycle_sec: float
    std_cycle_sec: float
    mean_sampling_sec: float
    std_sampling_sec: float
    mean_sampling_ms_per_step: float
    std_sampling_ms_per_step: float


@dataclass
class RunTimingSummary:
    experiment_name: str
    test_name: str
    seeds_start: int
    seeds_end: int
    run_suffix: str
    respacing: Optional[str] = None
    eta: Optional[str] = None
    batch_size: Optional[int] = None
    obs_guidance_mode: Optional[str] = None
    obs_guidance_space: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class TimingRecorder:
    def __init__(
        self,
        out_dir: Path,
        run_meta: RunTimingSummary,
        warmup_cycles: int = 0,
    ) -> None:
        self.out_dir = out_dir
        self.run_meta = run_meta
        self.warmup_cycles = max(0, int(warmup_cycles))
        self.cycle_rows: List[CycleTimingRow] = []
        self.seed_summaries: List[SeedTimingSummary] = []

    def add_cycle_row(self, row: CycleTimingRow) -> None:
        self.cycle_rows.append(row)

    def add_seed_summary(
        self,
        seed: int,
        total_sec: float,
        rows_for_seed: Sequence[CycleTimingRow],
    ) -> None:
        rows_for_seed = list(rows_for_seed)
        num_cycles = len(rows_for_seed)
        kept = rows_for_seed[self.warmup_cycles :]
        cycle_times = [r.cycle_total_sec for r in kept]
        sampling_times = [r.sampling_sec for r in kept]
        ms_per_step = [r.sampling_ms_per_step for r in kept]

        self.seed_summaries.append(
            SeedTimingSummary(
                seed=int(seed),
                total_sec=float(total_sec),
                num_cycles=int(num_cycles),
                warmup_cycles=int(self.warmup_cycles),
                mean_cycle_sec=_safe_mean(cycle_times),
                std_cycle_sec=_safe_std(cycle_times),
                mean_sampling_sec=_safe_mean(sampling_times),
                std_sampling_sec=_safe_std(sampling_times),
                mean_sampling_ms_per_step=_safe_mean(ms_per_step),
                std_sampling_ms_per_step=_safe_std(ms_per_step),
            )
        )

    def write(self) -> None:
        # Write per-cycle CSV
        cycle_csv = self.out_dir / "timing_cycles.csv"
        cycle_fields = [
            "seed",
            "i_cycle",
            "cycle_index",
            "sampling_steps",
            "cycle_total_sec",
            "sampling_sec",
            "sampling_ms_per_step",
            "lr_forecast_sec",
        ]
        write_csv(
            cycle_csv,
            (asdict(r) for r in self.cycle_rows),
            fieldnames=cycle_fields,
        )

        # Write per-seed summary CSV
        seed_csv = self.out_dir / "timing_seed_summary.csv"
        seed_fields = [
            "seed",
            "total_sec",
            "num_cycles",
            "warmup_cycles",
            "mean_cycle_sec",
            "std_cycle_sec",
            "mean_sampling_sec",
            "std_sampling_sec",
            "mean_sampling_ms_per_step",
            "std_sampling_ms_per_step",
        ]
        write_csv(
            seed_csv,
            (asdict(r) for r in self.seed_summaries),
            fieldnames=seed_fields,
        )

        # Write run metadata JSON (handy for plotting later)
        meta_json = self.out_dir / "timing_run_meta.json"
        payload = asdict(self.run_meta)
        payload["warmup_cycles"] = int(self.warmup_cycles)
        write_json(meta_json, payload)

