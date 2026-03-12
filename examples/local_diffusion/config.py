# Copyright (c) 2026 EarthBridge Team.
# Credits: Local Diffusion from Kim et al. "Tackling Structural Hallucination in Image Translation with Local Diffusion" ECCV 2024 Oral.

"""Configuration for Local Diffusion training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class LocalDiffusionConfig:
    """Configuration for Local Diffusion image-to-image translation.

    Parameters
    ----------
    dim : int
        Base channel dimension for the U-Net.
    channels : int
        Number of image channels (1 for grayscale, 3 for RGB).
    dim_mults : tuple of int
        Channel multipliers for each U-Net encoder/decoder level.
    resnet_block_groups : int
        Number of groups for GroupNorm in ResNet blocks.
    attn_dim_head : int
        Dimension per attention head.
    attn_heads : int
        Number of attention heads.
    full_attn : tuple of bool
        Per-level toggle for full vs. linear attention.
    cond_use_mid : bool
        Whether the condition encoder includes a mid-level block.
    num_train_timesteps : int
        Total number of training diffusion steps.
    beta_schedule : str
        Noise schedule: ``"linear"`` | ``"cosine"`` | ``"sigmoid"``.
    objective : str
        Prediction target: ``"pred_x0"`` | ``"pred_noise"`` | ``"pred_v"``.
    branch_out : bool
        Enable branch-and-fuse during inference.
    fusion_timestep : int
        Timestep to switch from branching to fusion.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    resolution : int
        Training image resolution.
    device : str
        Device to train on.
    save_dir : str
        Directory to save checkpoints.
    save_every : int
        Save checkpoint every N epochs.
    log_every : int
        Log metrics every N steps.
    """

    # Architecture
    dim: int = 32
    channels: int = 1
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8)
    resnet_block_groups: int = 8
    attn_dim_head: int = 32
    attn_heads: int = 4
    full_attn: Tuple[bool, ...] = (False, False, False, True)
    cond_in_channels: Optional[int] = None
    cond_filters: Optional[Tuple[int, ...]] = None
    init_type: str = "normal"
    init_gain: float = 0.02

    # Diffusion
    num_train_timesteps: int = 250
    beta_schedule: str = "sigmoid"
    objective: str = "pred_x0"
    loss_type: str = "min_snr"  # mse | min_snr | sid2 | edm
    min_snr_loss_weight: bool = False
    min_snr_gamma: float = 5.0

    # Inference
    branch_out: bool = True
    fusion_timestep: int = 2
    ddim_sampling_eta: float = 0.0

    # Training
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    max_grad_norm: float = 1.0
    epochs: int = 200
    batch_size: int = 64
    resolution: int = 28

    # Paths
    device: str = "cuda"
    save_dir: str = "./checkpoints/local_diffusion"
    save_every: int = 10
    log_every: int = 100
    resume_from: Optional[str] = None
