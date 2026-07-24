# Copyright (c) 2026 EarthBridge Team.
"""Configuration for SPADE training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class SPADEConfig:
    """Configuration for SPADE semantic image synthesis."""

    # Architecture
    label_nc: int = 35
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    n_layers_D: int = 4
    num_D: int = 2
    crop_size: int = 256
    aspect_ratio: float = 1.0
    num_upsampling_layers: Literal["normal", "more", "most"] = "normal"
    use_vae: bool = False
    z_dim: int = 256
    use_spectral_norm: bool = True
    param_free_norm_type: str = "instance"

    # Training
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.0
    beta2: float = 0.9
    lambda_gan: float = 1.0
    lambda_feat: float = 10.0
    lambda_perceptual: float = 10.0
    lambda_kl: float = 0.05
    gan_mode: str = "hinge"

    epochs: int = 200
    batch_size: int = 4
    resolution: int = 256

    device: str = "cuda"
    save_dir: str = "./checkpoints/spade"
    save_every: int = 10
    log_every: int = 100
    resume_from: Optional[str] = None
