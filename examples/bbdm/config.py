# Copyright (c) 2026 EarthBridge Team.
# Credits: BBDM from Li et al. "BBDM: Image-to-image Diffusion with Brownian Bridge" CVPR 2023.

"""Configuration for BBDM training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BBDMConfig:
    """Configuration for BBDM paired image-to-image translation.

    BBDM learns a Brownian bridge from target x0 to source y, then samples
    from y back to x0 via the trained reverse process.
    """

    # Architecture
    image_size: int = 256
    in_channels: int = 3
    model_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (1,)
    dropout: float = 0.0
    condition_mode: str = "concat"

    # Scheduler
    num_timesteps: int = 1000
    mt_type: str = "linear"
    objective: str = "grad"  # grad | noise | ysubx

    # Training
    optimizer: str = "prodigy"  # prodigy | adamw | adam | muon
    lr: float = 1.0  # 1.0 for Prodigy; 1e-4 for AdamW
    weight_decay: float = 0.01
    prodigy_d0: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.999
    epochs: int = 200
    batch_size: int = 16
    resolution: int = 256

    device: str = "cuda"
    save_dir: str = "./checkpoints/bbdm"
    save_every: int = 10
    log_every: int = 100
    resume_from: Optional[str] = None
