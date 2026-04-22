from __future__ import annotations

import os

import torch
import torch.distributed as dist


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_ddp() -> bool:
    return get_world_size() > 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_ddp(backend: str = "nccl") -> None:
    if not is_ddp():
        return
    # Set device early so NCCL binds to the correct GPU.
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return value
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= get_world_size()
    return value
