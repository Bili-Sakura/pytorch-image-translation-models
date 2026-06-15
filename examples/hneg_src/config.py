# Copyright (c) 2026 EarthBridge Team.
# Credits: Hneg-SRC from Jung et al. CVPR 2022 — https://github.com/jcy132/Hneg_SRC

"""Configuration for Hneg-SRC unpaired image-to-image translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HnegSRCConfig:
    """Configuration for Hneg-SRC (Semantic Relation Contrastive) training.

    Extends the CUT backbone with SRC and HDCE losses from
    `Jung et al., CVPR 2022 <https://arxiv.org/abs/2203.01532>`_.
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

    lambda_GAN: float = 1.0
    lambda_HDCE: float = 0.1
    lambda_SRC: float = 0.05
    dce_idt: bool = True
    nce_layers: str = "0,4,8,12,16"
    nce_T: float = 0.07
    num_patches: int = 256
    nce_includes_all_negatives_from_minibatch: bool = False
    gan_mode: str = "lsgan"

    use_curriculum: bool = True
    hdce_gamma: float = 50.0
    hdce_gamma_min: float = 10.0
    step_gamma: bool = False
    step_gamma_epoch: int = 200
    no_hneg: bool = False

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
    save_dir: str = "./checkpoints/hneg_src"
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
