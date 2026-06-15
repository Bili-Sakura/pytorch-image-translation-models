# Copyright (c) 2026 EarthBridge Team.
# Credits: Decent from Xie et al. NeurIPS 2022 — https://github.com/Mid-Push/Decent

"""Configuration for Decent unpaired image-to-image translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DecentConfig:
    """Configuration for Decent (Density Changing Regularization) training.

    Extends the CUT ResNet generator backbone with per-domain normalizing-flow
    density estimators and a density-changing variance loss from
    `Xie et al., NeurIPS 2022 <https://github.com/Mid-Push/Decent>`_.
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
    no_antialias: bool = True
    no_antialias_up: bool = True
    init_type: str = "xavier"
    init_gain: float = 0.02

    lambda_GAN: float = 1.0
    lambda_var: float = 0.01
    lambda_idt: float = 10.0
    var_layers: str = "0,4,8,12,16"
    num_patches: int = 256
    var_all: bool = False

    flow_type: str = "bnaf"
    flow_blocks: int = 1
    bnaf_layers: int = 0
    bnaf_dim: int = 10
    maf_dim: int = 1024
    maf_layers: int = 2
    maf_comps: int = 10
    flow_lr: float = 1e-3
    flow_ema: float = 0.998
    gan_mode: str = "lsgan"

    lr: float = 2e-4
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    prodigy_d0: float = 1e-6
    beta1: float = 0.5
    beta2: float = 0.999
    epochs: int = 200
    n_epochs_decay: int = 200
    batch_size: int = 4
    resolution: int = 256

    device: str = "cuda"
    save_dir: str = "./checkpoints/decent"
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
