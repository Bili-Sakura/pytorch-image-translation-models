# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""CycleDiff training and inference configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def packaged_configs_dir() -> Path:
    """Default YAML configs shipped under ``examples/cyclediff/configs``."""
    return Path(__file__).resolve().parent / "configs"


@dataclass
class CycleDiffConfig:
    """Paths for CycleDiff training or translation."""

    cfg: str
    ckpt_path: Optional[str] = None
    device: str = "cuda"
    results_folder: Optional[str] = None
