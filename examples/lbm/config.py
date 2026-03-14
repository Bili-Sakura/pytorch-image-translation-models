# Copyright (c) 2026 EarthBridge Team.
"""Configuration for LBM training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LBMConfig:
    """Configuration for LBM (Latent Bridge Matching)."""

    image_size: int = 256
    in_channels: int = 3
    model_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (1,)
    dropout: float = 0.0
    condition_mode: str = "concat"

    num_train_timesteps: int = 1000
    bridge_noise_sigma: float = 0.001

    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epochs: int = 200
    batch_size: int = 16
    resolution: int = 256

    device: str = "cuda"
    save_dir: str = "./checkpoints/lbm"
    save_every: int = 10
    log_every: int = 100
    resume_from: Optional[str] = None
