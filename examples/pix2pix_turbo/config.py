# Copyright (c) 2026 EarthBridge Team.
# Credits: img2img-turbo from Parmar et al. 2024 —
# https://github.com/GaParmar/img2img-turbo

"""Configuration for pix2pix-turbo paired training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Pix2PixTurboConfig:
    """Training configuration for pix2pix-turbo."""

    dataset_folder: str = ""
    train_image_prep: str = "resized_crop_512"
    test_image_prep: str = "resized_crop_512"
    output_dir: str = "./checkpoints/pix2pix_turbo"
    tracker_project_name: str = "pix2pix_turbo"

    lora_rank_unet: int = 8
    lora_rank_vae: int = 4
    base_model: str = "stabilityai/sd-turbo"

    lambda_gan: float = 0.5
    lambda_lpips: float = 5.0
    lambda_l2: float = 1.0
    lambda_clipsim: float = 5.0
    gan_disc_type: str = "vagan_clip"
    gan_loss_type: str = "multilevel_sigmoid_s"

    learning_rate: float = 5e-6
    train_batch_size: int = 4
    num_training_epochs: int = 10
    max_train_steps: int = 10_000
    gradient_accumulation_steps: int = 1
    checkpointing_steps: int = 500
    eval_freq: int = 100
    viz_freq: int = 100
    num_samples_eval: int = 100
    track_val_fid: bool = False
    resolution: int = 512

    seed: Optional[int] = 42
    device: str = "cuda"
    report_to: str = "wandb"
    mixed_precision: Optional[str] = None
    enable_xformers_memory_efficient_attention: bool = False
    gradient_checkpointing: bool = False
    allow_tf32: bool = False
    set_grads_to_none: bool = False
    dataloader_num_workers: int = 0

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
