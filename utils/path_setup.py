"""Utilities to ensure repo and legacy python paths are importable."""

from __future__ import annotations

import sys
from pathlib import Path


def setup_paths():
    """Add repo root and python/ subdir to sys.path if missing."""
    here = Path().resolve()
    root = here
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            root = p
            break
    for candidate in (root, root / "python"):
        candidate = candidate.resolve()
        if str(candidate) not in sys.path:
            sys.path.append(str(candidate))

    return root
