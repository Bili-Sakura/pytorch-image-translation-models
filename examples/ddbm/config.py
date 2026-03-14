# Copyright (c) 2026 EarthBridge Team.
"""Configuration for DDBM training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DDBMConfig:
    """Configuration for DDBM image-to-image translation."""

    image_size: int = 256
    in_channels: int = 3
    model_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (1,)
    dropout: float = 0.0
    condition_mode: str = "concat"

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5
    rho: float = 7.0
    num_train_timesteps: int = 40

    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epochs: int = 200
    batch_size: int = 16
    resolution: int = 256

    device: str = "cuda"
    save_dir: str = "./checkpoints/ddbm"
    save_every: int = 10
    log_every: int = 100
    resume_from: Optional[str] = None
