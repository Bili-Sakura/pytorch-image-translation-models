# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""Path resolution for a local CycleDiff repository checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from examples.community.cyclediff.model import CYCLEDIFF_REPO_URL


def _is_cyclediff_root(p: Path) -> bool:
    if not p.is_dir():
        return False
    markers = (
        p / "train_uncond_ldm_cycle.py",
        p / "translation_uncond_ldm_cycle.py",
        p / "ddm" / "__init__.py",
    )
    return all(m.is_file() for m in markers)


def resolve_cyclediff_root(cyclediff_src_path: Optional[str | Path] = None) -> Path:
    """Return the root directory of a CycleDiff clone.

    Parameters
    ----------
    cyclediff_src_path : str | Path, optional
        Explicit path to the CycleDiff repository root. When omitted, the
        ``CYCLEDIFF_ROOT`` environment variable is checked, then common locations
        next to this monorepo or the current working directory.

    Returns
    -------
    Path
        Resolved absolute path to the CycleDiff checkout.

    Raises
    ------
    FileNotFoundError
        If no valid CycleDiff root can be found.
    """
    if cyclediff_src_path is not None:
        p = Path(cyclediff_src_path).expanduser()
        if _is_cyclediff_root(p):
            return p.resolve()
        raise FileNotFoundError(
            f"CycleDiff source not found or incomplete at {p}. "
            "Expected train_uncond_ldm_cycle.py, translation_uncond_ldm_cycle.py, and ddm/__init__.py."
        )

    env = os.environ.get("CYCLEDIFF_ROOT")
    if env:
        p = Path(env).expanduser()
        if _is_cyclediff_root(p):
            return p.resolve()
        raise FileNotFoundError(
            f"CYCLEDIFF_ROOT is set to {env!r} but that path is not a valid CycleDiff checkout."
        )

    root = Path(__file__).resolve().parents[4]
    candidates = [
        root / "CycleDiff",
        root / "projects" / "CycleDiff",
        root / "external" / "CycleDiff",
        Path.cwd() / "CycleDiff",
        Path.cwd() / "projects" / "CycleDiff",
    ]
    for p in candidates:
        if _is_cyclediff_root(p):
            return p.resolve()

    raise FileNotFoundError(
        f"CycleDiff source not found. Clone {CYCLEDIFF_REPO_URL} and set CYCLEDIFF_ROOT or pass "
        "cyclediff_src_path, or place the repo at ./CycleDiff, ./projects/CycleDiff, or ./external/CycleDiff."
    )


def inject_cyclediff_sys_path(cyclediff_src_path: Optional[str | Path] = None) -> Path:
    """Prepend the CycleDiff root to ``sys.path`` so ``import ddm`` works from upstream code.

    Returns
    -------
    Path
        The resolved CycleDiff root.
    """
    resolved = resolve_cyclediff_root(cyclediff_src_path)
    s = str(resolved)
    if s not in sys.path:
        sys.path.insert(0, s)
    return resolved


__all__ = ["inject_cyclediff_sys_path", "resolve_cyclediff_root"]
