# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiffusion (Wu & De la Torre, ICCV 2023) — https://github.com/humansensinglab/cycle-diffusion

"""Path resolution for a local humansensinglab/cycle-diffusion repository checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from examples.community.cycle_diffusion.model import CYCLE_DIFFUSION_REPO_URL


def _is_cycle_diffusion_root(p: Path) -> bool:
    if not p.is_dir():
        return False
    markers = (
        p / "main.py",
        p / "trainer" / "trainer.py",
        p / "utils" / "config_utils.py",
        p / "model" / "__init__.py",
    )
    return all(m.is_file() for m in markers)


def resolve_cycle_diffusion_root(src_path: Optional[str | Path] = None) -> Path:
    """Return the root directory of a cycle-diffusion clone.

    Parameters
    ----------
    src_path : str | Path, optional
        Explicit path to the repository root. When omitted, the
        ``CYCLE_DIFFUSION_ROOT`` environment variable is checked, then common
        locations next to this monorepo or the current working directory.

    Returns
    -------
    Path
        Resolved absolute path to the checkout.

    Raises
    ------
    FileNotFoundError
        If no valid root can be found.
    """
    if src_path is not None:
        p = Path(src_path).expanduser()
        if _is_cycle_diffusion_root(p):
            return p.resolve()
        raise FileNotFoundError(
            f"cycle-diffusion source not found or incomplete at {p}. "
            "Expected main.py, trainer/trainer.py, utils/config_utils.py, and model/__init__.py."
        )

    env = os.environ.get("CYCLE_DIFFUSION_ROOT")
    if env:
        p = Path(env).expanduser()
        if _is_cycle_diffusion_root(p):
            return p.resolve()
        raise FileNotFoundError(
            f"CYCLE_DIFFUSION_ROOT is set to {env!r} but that path is not a valid cycle-diffusion checkout."
        )

    root = Path(__file__).resolve().parents[4]
    candidates = [
        root / "cycle-diffusion",
        root / "projects" / "cycle-diffusion",
        root / "external" / "cycle-diffusion",
        Path.cwd() / "cycle-diffusion",
        Path.cwd() / "projects" / "cycle-diffusion",
    ]
    for p in candidates:
        if _is_cycle_diffusion_root(p):
            return p.resolve()

    raise FileNotFoundError(
        f"cycle-diffusion source not found. Clone {CYCLE_DIFFUSION_REPO_URL} and set CYCLE_DIFFUSION_ROOT or pass "
        "src_path, or place the repo at ./cycle-diffusion, ./projects/cycle-diffusion, or ./external/cycle-diffusion."
    )


def inject_cycle_diffusion_sys_path(src_path: Optional[str | Path] = None) -> Path:
    """Prepend the cycle-diffusion root to ``sys.path`` for imports of ``model``, ``trainer``, etc.

    Returns
    -------
    Path
        The resolved repository root.
    """
    resolved = resolve_cycle_diffusion_root(src_path)
    s = str(resolved)
    if s not in sys.path:
        sys.path.insert(0, s)
    return resolved


__all__ = ["inject_cycle_diffusion_sys_path", "resolve_cycle_diffusion_root"]
