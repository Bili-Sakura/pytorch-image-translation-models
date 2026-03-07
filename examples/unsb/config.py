# Copyright (c) 2026 EarthBridge Team.
# Credits: UNSB from Kim et al. "Unpaired Image-to-Image Translation via Neural Schrödinger Bridge" ICLR 2024.

"""Configuration for UNSB (Unpaired Neural Schrödinger Bridge) training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class UNSBConfig:
    """Configuration for UNSB image-to-image translation.

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
    n_blocks : int
        Number of conditional ResNet blocks in the generator.
    n_mlp : int
        Number of MLP layers in the noise mapping network.
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
    lambda_SB : float
        Weight for Schrödinger Bridge loss.
    tau : float
        Entropy parameter for SDE bridge dynamics.
    num_timesteps : int
        Number of bridge refinement steps (NFE).
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
        Learning rate for G, D, E.
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
    n_blocks: int = 9
    n_mlp: int = 3
    n_layers_D: int = 3
    normG: str = "instance"
    normD: str = "instance"
    use_dropout: bool = False
    no_antialias: bool = False
    no_antialias_up: bool = False
    init_type: str = "normal"
    init_gain: float = 0.02

    # Feature network (PatchSampleMLP)
    netF: str = "mlp_sample"
    netF_nc: int = 256

    # Loss weights
    lambda_GAN: float = 1.0
    lambda_NCE: float = 1.0
    lambda_SB: float = 1.0
    tau: float = 0.01
    num_timesteps: int = 5
    nce_idt: bool = True
    nce_layers: str = "0,4,8,12,16"
    nce_T: float = 0.07
    num_patches: int = 256
    nce_includes_all_negatives_from_minibatch: bool = False
    gan_mode: str = "lsgan"

    # Training
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    epochs: int = 200
    batch_size: int = 1
    resolution: int = 256

    # Paths
    device: str = "cuda"
    save_dir: str = "./checkpoints/unsb"
    save_every: int = 10
    log_every: int = 100
    resume_from: Optional[str] = None
