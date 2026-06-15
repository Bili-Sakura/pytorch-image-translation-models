# Copyright (c) 2026 EarthBridge Team.
# Credits: F-LSeSim from Zheng et al. CVPR 2021 —
# https://github.com/lyndonzheng/F-LSeSim

"""Configuration for F-LSeSim unpaired image-to-image translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FLSeSimConfig:
    """Configuration for F-LSeSim (Spatially-Correlative Loss) training.

    Based on `Zheng et al., CVPR 2021
    <https://arxiv.org/abs/2104.00209>`_ and the upstream
    `F-LSeSim <https://github.com/lyndonzheng/F-LSeSim>`_ repository.
    """

    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    netG: str = "resnet_9blocks"
    netD: str = "basic"
    n_layers_D: int = 3
    normG: str = "instance"
    normD: str = "instance"
    no_dropout: bool = True
    no_antialias: bool = False
    no_antialias_up: bool = False
    init_type: str = "normal"
    init_gain: float = 0.02

    lambda_GAN: float = 1.0
    lambda_spatial: float = 10.0
    lambda_spatial_idt: float = 0.0
    lambda_identity: float = 0.0
    lambda_perceptual: float = 0.0
    lambda_style: float = 0.0
    lambda_gradient: float = 0.0

    attn_layers: str = "4,7,9"
    patch_nums: int = 256
    patch_size: int = 64
    loss_mode: str = "cos"
    use_norm: bool = False
    learned_attn: bool = False
    temperature: float = 0.07
    gan_mode: str = "lsgan"
    pool_size: int = 50

    lr: float = 2e-4
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    prodigy_d0: float = 1e-6
    beta1: float = 0.5
    beta2: float = 0.999
    epochs: int = 200
    n_epochs_decay: int = 200
    batch_size: int = 1
    resolution: int = 256

    device: str = "cuda"
    save_dir: str = "./checkpoints/flsesim"
    save_every: int = 10
    log_every: int = 100
    resume_from: Optional[str] = None

    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    checkpointing_steps: Optional[int] = None
    checkpoints_total_limit: int = 1
    validation_steps: Optional[int] = None
    mixed_precision: str = "no"
    dataloader_num_workers: int = 0
    seed: Optional[int] = 42
    output_dir: Optional[str] = None
