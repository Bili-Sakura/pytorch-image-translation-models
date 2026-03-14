# Copyright (c) 2026 EarthBridge Team.
# Credits: Berman et al., NeurIPS 2025 - Bosch Research LDDBM.

"""Configuration for LDDBM training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LDDBMConfig:
    """Configuration for LDDBM (Latent Diffusion Bridge Model) training.

    For super-resolution 16→128: source=LQ 16×16, target=HQ 128×128.
    """

    resolution_source: int = 16
    resolution_target: int = 128

    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epochs: int = 200
    batch_size: int = 16

    device: str = "cuda"
    save_dir: str = "./checkpoints/lddbm"
    save_every: int = 10
    log_every: int = 100
    resume_from: Optional[str] = None
