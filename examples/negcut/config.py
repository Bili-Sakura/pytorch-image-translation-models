# Copyright (c) 2026 EarthBridge Team.
# Credits: NEGCUT from Wang et al. ICCV 2021 — https://github.com/WeilunWang/NEGCUT

"""Configuration for NEGCUT unpaired image-to-image translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class NEGCUTConfig:
    """Configuration for NEGCUT (hard-negative contrastive) training.

    Extends the CUT backbone with an adversarial negative generator and
    learned PatchNCE loss from
    `Wang et al., ICCV 2021 <https://github.com/WeilunWang/NEGCUT>`_.
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

    netF: str = "mlp_sample"
    netF_nc: int = 256
    netN: str = "neg_gen_momentum"

    negcut_mode: str = "negcut"
    lambda_GAN: float = 1.0
    lambda_NCE: float = 1.0
    lambda_MS_neg: float = 1.0
    nce_idt: bool = True
    nce_layers: str = "0,4,8,12,16"
    nce_T: float = 0.07
    num_patches: int = 256
    nce_includes_all_negatives_from_minibatch: bool = False
    gan_mode: str = "lsgan"
    momentum_decay: float = 0.9

    lr: float = 2e-4
    lr_N: float = 2e-4
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    prodigy_d0: float = 1e-6
    beta1: float = 0.5
    beta2: float = 0.999
    epochs: int = 200
    batch_size: int = 4
    resolution: int = 256

    device: str = "cuda"
    save_dir: str = "./checkpoints/negcut"
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
