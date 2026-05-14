# Copyright (c) 2026 EarthBridge Team.
# Credits: SDEdit (Meng et al., ICLR 2022) — https://github.com/ermongroup/SDEdit

"""Path resolution for a local ermongroup/SDEdit repository checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from examples.community.sdedit.model import SDEDIT_REPO_URL


def _is_sdedit_root(p: Path) -> bool:
    if not p.is_dir():
        return False
    markers = (
        p / "main.py",
        p / "runners" / "image_editing.py",
        p / "models" / "diffusion.py",
        p / "configs",
    )
    if not all(m.exists() for m in markers):
        return False
    if not (p / "configs").is_dir():
        return False
    return any((p / "configs").glob("*.yml"))


def resolve_sdedit_root(src_path: Optional[str | Path] = None) -> Path:
    """Return the root directory of an SDEdit clone.

    Parameters
    ----------
    src_path : str | Path, optional
        Explicit path to the repository root. When omitted, the
        ``SDEDIT_ROOT`` environment variable is checked, then common
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
        if _is_sdedit_root(p):
            return p.resolve()
        raise FileNotFoundError(
            f"SDEdit source not found or incomplete at {p}. "
            "Expected main.py, runners/image_editing.py, models/diffusion.py, and configs/*.yml."
        )

    env = os.environ.get("SDEDIT_ROOT")
    if env:
        p = Path(env).expanduser()
        if _is_sdedit_root(p):
            return p.resolve()
        raise FileNotFoundError(
            f"SDEDIT_ROOT is set to {env!r} but that path is not a valid SDEdit checkout."
        )

    root = Path(__file__).resolve().parents[4]
    candidates = [
        root / "SDEdit",
        root / "sdedit",
        root / "projects" / "SDEdit",
        root / "external" / "SDEdit",
        Path.cwd() / "SDEdit",
        Path.cwd() / "sdedit",
        Path.cwd() / "projects" / "SDEdit",
    ]
    for p in candidates:
        if _is_sdedit_root(p):
            return p.resolve()

    raise FileNotFoundError(
        f"SDEdit source not found. Clone {SDEDIT_REPO_URL} and set SDEDIT_ROOT or pass "
        "src_path, or place the repo at ./SDEdit, ./sdedit, ./projects/SDEdit, or ./external/SDEdit."
    )


def inject_sdedit_sys_path(src_path: Optional[str | Path] = None) -> Path:
    """Prepend the SDEdit root to ``sys.path`` for imports of ``runners``, ``models``, etc.

    Returns
    -------
    Path
        The resolved repository root.
    """
    resolved = resolve_sdedit_root(src_path)
    s = str(resolved)
    if s not in sys.path:
        sys.path.insert(0, s)
    return resolved


__all__ = ["inject_sdedit_sys_path", "resolve_sdedit_root"]
