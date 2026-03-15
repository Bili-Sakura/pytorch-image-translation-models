# Copyright (c) 2026 EarthBridge Team.
# Credits: CUT from Park et al. ECCV 2020.

"""Configuration for CUT (Contrastive Unpaired Translation) training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CUTConfig:
    """Configuration for CUT image-to-image translation.

    Parameters
    ----------
    input_nc : int
        Number of input (source domain) channels.
    output_nc : int
        Number of output (target domain) channels.
    ngf : int
        Base number of generator filters.
    ndf : int
        Base number of discriminator filters.
    netG : str
        Generator architecture: ``resnet_9blocks`` | ``resnet_6blocks`` | ``resnet_4blocks``.
    netD : str
        Discriminator type (``basic`` = 70×70 PatchGAN).
    n_layers_D : int
        Number of discriminator layers.
    normG : str
        Generator normalisation: ``instance`` | ``batch``.
    normD : str
        Discriminator normalisation.
    lambda_GAN : float
        Weight for GAN loss.
    lambda_NCE : float
        Weight for PatchNCE contrastive loss.
    nce_idt : bool
        Whether to use identity NCE loss.
    nce_layers : str
        Comma-separated layer indices for NCE feature extraction.
    nce_T : float
        Temperature for contrastive loss.
    num_patches : int
        Number of patches to sample for NCE.
    gan_mode : str
        GAN loss: ``lsgan`` | ``vanilla`` | ``wgangp``.
    lr : float
        Learning rate for G and D.
    beta1 : float
        Adam beta1.
    beta2 : float
        Adam beta2.
    epochs : int
        Number of training epochs.
    resolution : int
        Training resolution (crop/resize to this size).
    device : str
        Device to train on.
    save_dir : str
        Directory to save checkpoints.
    save_every : int
        Save checkpoint every N epochs.
    log_every : int
        Log metrics every N steps.
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

    # Feature network (PatchSampleMLP)
    netF: str = "mlp_sample"
    netF_nc: int = 256

    # Loss weights
    lambda_GAN: float = 1.0
    lambda_NCE: float = 1.0
    nce_idt: bool = True
    nce_layers: str = "0,4,8,12,16"
    nce_T: float = 0.07
    num_patches: int = 256
    nce_includes_all_negatives_from_minibatch: bool = False
    gan_mode: str = "lsgan"

    # Training
    lr: float = 2e-4
    optimizer: str = "adamw"  # "adamw" | "adam" | "prodigy" | "muon"
    weight_decay: float = 0.01
    prodigy_d0: float = 1e-6
    beta1: float = 0.5
    beta2: float = 0.999
    epochs: int = 200
    batch_size: int = 4
    resolution: int = 256

    # Paths
    device: str = "cuda"
    save_dir: str = "./checkpoints/cut"
    save_every: int = 10
    log_every: int = 100
    resume_from: Optional[str] = None

    # Extended training (4th-MAVIC-T style)
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    checkpointing_steps: Optional[int] = None
    checkpoints_total_limit: int = 1
    validation_steps: Optional[int] = None
    mixed_precision: str = "no"
    dataloader_num_workers: int = 0
    seed: Optional[int] = 42

    # Alias: output_dir for 4th-MAVIC-T style scripts (save_dir takes precedence)
    output_dir: Optional[str] = None
