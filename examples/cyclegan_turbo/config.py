# Copyright (c) 2026 EarthBridge Team.
# Credits: img2img-turbo from Parmar et al. 2024 —
# https://github.com/GaParmar/img2img-turbo

"""Configuration for CycleGAN-Turbo unpaired training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CycleGANTurboConfig:
    """Training configuration for CycleGAN-Turbo."""

    dataset_folder: str = ""
    train_img_prep: str = "resize_512x512"
    val_img_prep: str = "resize_512x512"
    output_dir: str = "./checkpoints/cyclegan_turbo"
    tracker_project_name: str = "cyclegan_turbo"

    lora_rank_unet: int = 128
    lora_rank_vae: int = 4
    base_model: str = "stabilityai/sd-turbo"

    lambda_gan: float = 0.5
    lambda_idt: float = 1.0
    lambda_cycle: float = 1.0
    lambda_cycle_lpips: float = 10.0
    lambda_idt_lpips: float = 1.0
    gan_disc_type: str = "vagan_clip"
    gan_loss_type: str = "multilevel_sigmoid"

    learning_rate: float = 5e-6
    train_batch_size: int = 1
    max_train_epochs: int = 100
    max_train_steps: Optional[int] = 10_000
    gradient_accumulation_steps: int = 1
    checkpointing_steps: int = 500
    validation_steps: int = 500
    validation_num_images: int = -1
    viz_freq: int = 20

    seed: int = 42
    device: str = "cuda"
    report_to: str = "wandb"
    enable_xformers_memory_efficient_attention: bool = False
    gradient_checkpointing: bool = False
    allow_tf32: bool = False
    dataloader_num_workers: int = 0

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 10.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
