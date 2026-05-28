# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""Configuration for CycleDiff training and translation via upstream scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CycleDiffConfig:
    """Paths and defaults for delegating to a local CycleDiff checkout."""

    cyclediff_root: Optional[str] = None
    train_cfg: str = "./configs/cat2dog/translation_C_disc_timestep_ode_2.yaml"
    translation_cfg: str = "./configs/cat2dog/test_translation.yaml"
    vae_cfg: Optional[str] = None
    ldm_cfg: Optional[str] = None
