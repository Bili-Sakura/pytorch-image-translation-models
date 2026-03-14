# Copyright (c) 2026 EarthBridge Team.
"""Configuration for DDIB training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DDIBConfig:
    """Configuration for DDIB. Trains two unconditional diffusion models
    (source domain + target domain) for dual diffusion bridge translation."""

    image_size: int = 256
    in_channels: int = 3
    model_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (1,)
    dropout: float = 0.0

    num_train_timesteps: int = 1000
    noise_schedule: str = "linear"
    predict_xstart: bool = False

    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epochs: int = 200
    batch_size: int = 16
    resolution: int = 256

    device: str = "cuda"
    save_dir: str = "./checkpoints/ddib"
    save_every: int = 10
    log_every: int = 100
    resume_from: Optional[str] = None
