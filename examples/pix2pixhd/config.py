# Copyright (c) 2026 EarthBridge Team.
"""Configuration for Pix2PixHD training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Pix2PixHDConfig:
    """Configuration for Pix2PixHD paired image-to-image translation."""

    # Architecture
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    n_downsampling: int = 4
    n_blocks: int = 9
    ndf: int = 64
    n_layers_D: int = 3

    # Training
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    lambda_l1: float = 100.0
    lambda_perceptual: float = 10.0
    gan_mode: str = "lsgan"

    epochs: int = 200
    batch_size: int = 16
    resolution: int = 256

    device: str = "cuda"
    save_dir: str = "./checkpoints/pix2pixhd"
    save_every: int = 10
    log_every: int = 100
    resume_from: Optional[str] = None
