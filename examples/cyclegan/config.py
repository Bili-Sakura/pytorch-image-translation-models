# Credits: CycleGAN (Zhu et al., ICCV 2017) — https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""Configuration for CycleGAN unpaired image-to-image translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CycleGANConfig:
    """Training configuration for CycleGAN.

    Defaults follow the original
    `junyanz/pytorch-CycleGAN-and-pix2pix` repository.
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
    init_type: str = "normal"
    init_gain: float = 0.02

    lambda_A: float = 10.0
    lambda_B: float = 10.0
    lambda_identity: float = 0.5
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
    save_dir: str = "./checkpoints/cyclegan"
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
