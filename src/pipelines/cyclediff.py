# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""CycleDiff pipeline for unpaired image-to-image translation.

CycleDiff uses two latent diffusion models, cycle-consistent generators, and
adversarial discriminators in a local upstream checkout. This module resolves
that checkout, exposes a :class:`CycleDiffPipeline` facade, and can delegate
training or translation to upstream entry scripts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

from src.models.cyclediff import (
    CYCLEDIFF_LDM_TRAIN_SCRIPT,
    CYCLEDIFF_REPO_URL,
    CYCLEDIFF_TRAIN_SCRIPT,
    CYCLEDIFF_TRANSLATION_SCRIPT,
    CYCLEDIFF_VAE_TRAIN_SCRIPT,
)


def _repo_root() -> Path:
    """Monorepo root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def _is_cyclediff_root(p: Path) -> bool:
    if not p.is_dir():
        return False
    markers = (
        p / CYCLEDIFF_TRAIN_SCRIPT,
        p / CYCLEDIFF_TRANSLATION_SCRIPT,
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
            f"Expected {CYCLEDIFF_TRAIN_SCRIPT}, {CYCLEDIFF_TRANSLATION_SCRIPT}, and ddm/__init__.py."
        )

    env = os.environ.get("CYCLEDIFF_ROOT")
    if env:
        p = Path(env).expanduser()
        if _is_cyclediff_root(p):
            return p.resolve()
        raise FileNotFoundError(
            f"CYCLEDIFF_ROOT is set to {env!r} but that path is not a valid CycleDiff checkout."
        )

    root = _repo_root()
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
    """Prepend the CycleDiff root to ``sys.path`` so ``import ddm`` works.

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


def run_cyclediff_script(
    script_and_args: Sequence[str],
    *,
    cyclediff_root: Optional[str | Path] = None,
) -> int:
    """Run an upstream CycleDiff script with cwd set to the checkout root.

    Parameters
    ----------
    script_and_args : sequence of str
        Script name (e.g. ``train_uncond_ldm_cycle.py``) followed by CLI arguments.
    cyclediff_root : str | Path, optional
        Path to CycleDiff clone (overrides auto-discovery).

    Returns
    -------
    int
        Subprocess exit code.
    """
    if not script_and_args:
        raise ValueError("script_and_args must include at least the script name")

    script_name = script_and_args[0]
    script_args = list(script_and_args[1:])
    root = resolve_cyclediff_root(cyclediff_root)
    script_path = root / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found under CycleDiff root: {script_path}")

    cmd = [sys.executable, str(script_path), *script_args]
    return subprocess.call(cmd, cwd=str(root))


@dataclass
class CycleDiffPipelineOutput:
    """Placeholder output when running batch translation via upstream scripts.

    Upstream ``translation_uncond_ldm_cycle.py`` writes images to ``sampler.save_folder``
    on disk; this dataclass is used when wrapping programmatic inference in the future.
    """

    images: Optional[List] = None
    results_folder: Optional[Path] = None


class CycleDiffPipeline:
    """Facade for CycleDiff training and inference via a local upstream checkout.

    Example
    -------
    ::

        from src.pipelines.cyclediff import CycleDiffPipeline

        pipe = CycleDiffPipeline.from_pretrained(cyclediff_root="/path/to/CycleDiff")
        pipe.run_training(cfg="./configs/cat2dog/translation_C_disc_timestep_ode_2.yaml")
        pipe.run_translation(cfg="./configs/cat2dog/test_translation.yaml")
    """

    def __init__(self, cyclediff_root: str | Path) -> None:
        self.cyclediff_root = resolve_cyclediff_root(cyclediff_root)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str | Path] = None,
        *,
        cyclediff_root: Optional[str | Path] = None,
        **kwargs,
    ) -> "CycleDiffPipeline":
        """Create a pipeline bound to a CycleDiff checkout.

        Parameters
        ----------
        pretrained_model_name_or_path : str | Path, optional
            Alias for ``cyclediff_root`` (diffusers-style naming). When both are
            given, ``cyclediff_root`` takes precedence.
        cyclediff_root : str | Path, optional
            Path to the CycleDiff repository root.
        """
        root = cyclediff_root
        if root is None and pretrained_model_name_or_path is not None:
            root = pretrained_model_name_or_path
        if root is None:
            root = resolve_cyclediff_root()
        return cls(cyclediff_root=root)

    def inject_upstream_path(self) -> Path:
        """Add the checkout to ``sys.path`` and return its path."""
        return inject_cyclediff_sys_path(self.cyclediff_root)

    def run_training(
        self,
        cfg: str | Path,
        *,
        script: str = CYCLEDIFF_TRAIN_SCRIPT,
        extra_args: Optional[Sequence[str]] = None,
    ) -> int:
        """Run cycle LDM training (``train_uncond_ldm_cycle.py`` by default)."""
        args = ["--cfg", str(cfg)]
        if extra_args:
            args.extend(extra_args)
        return run_cyclediff_script([script, *args], cyclediff_root=self.cyclediff_root)

    def run_vae_training(
        self,
        cfg: str | Path,
        *,
        extra_args: Optional[Sequence[str]] = None,
    ) -> int:
        """Run VAE pretraining (``train_vae.py``)."""
        args = ["--cfg", str(cfg)]
        if extra_args:
            args.extend(extra_args)
        return run_cyclediff_script(
            [CYCLEDIFF_VAE_TRAIN_SCRIPT, *args], cyclediff_root=self.cyclediff_root
        )

    def run_ldm_training(
        self,
        cfg: str | Path,
        *,
        extra_args: Optional[Sequence[str]] = None,
    ) -> int:
        """Run single-domain LDM pretraining (``train_uncond_ldm.py``)."""
        args = ["--cfg", str(cfg)]
        if extra_args:
            args.extend(extra_args)
        return run_cyclediff_script(
            [CYCLEDIFF_LDM_TRAIN_SCRIPT, *args], cyclediff_root=self.cyclediff_root
        )

    def run_translation(
        self,
        cfg: str | Path,
        *,
        extra_args: Optional[Sequence[str]] = None,
    ) -> int:
        """Run unpaired translation inference (``translation_uncond_ldm_cycle.py``)."""
        args = ["--cfg", str(cfg)]
        if extra_args:
            args.extend(extra_args)
        return run_cyclediff_script(
            [CYCLEDIFF_TRANSLATION_SCRIPT, *args], cyclediff_root=self.cyclediff_root
        )

    def run_script(
        self,
        script: str,
        script_args: Optional[Sequence[str]] = None,
    ) -> int:
        """Run any script from the CycleDiff root."""
        argv = [script]
        if script_args:
            argv.extend(script_args)
        return run_cyclediff_script(argv, cyclediff_root=self.cyclediff_root)


def load_cyclediff_pipeline(
    cyclediff_root: Optional[str | Path] = None,
) -> CycleDiffPipeline:
    """Load a :class:`CycleDiffPipeline` bound to a local CycleDiff checkout."""
    return CycleDiffPipeline.from_pretrained(cyclediff_root=cyclediff_root)


__all__ = [
    "CYCLEDIFF_REPO_URL",
    "CycleDiffPipeline",
    "CycleDiffPipelineOutput",
    "inject_cyclediff_sys_path",
    "load_cyclediff_pipeline",
    "resolve_cyclediff_root",
    "run_cyclediff_script",
]
