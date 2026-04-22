from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_experiment_output_root(
    root_dir: str,
    experiment_name: str,
    processed_base_dir: Optional[str] = None,
) -> Path:
    """
    Return the base directory where processed SRDA outputs for a given experiment
    should live.

    If processed_base_dir is provided it is interpreted as either an absolute path
    or relative to root_dir and used directly. Otherwise the legacy
    data/srda/processed/<experiment> directory is used.
    """
    if processed_base_dir:
        base = Path(processed_base_dir)
        if not base.is_absolute():
            base = Path(root_dir) / base
    else:
        base = Path(root_dir) / "data" / "srda" / "processed" / experiment_name
    return base
