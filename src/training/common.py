"""Common training configuration fields shared across tutorial trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseTrainingConfig:
    """Base configuration for tutorial training scripts.

    Mirrors common fields from 4th-MAVIC-T style scripts:
    - Step-based training (max_train_steps, checkpointing_steps)
    - Checkpoint rotation (checkpoints_total_limit)
    - Mixed precision, gradient accumulation
    - Logging (log_with, validation_steps)
    """

    # Output and resume
    output_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None  # "latest" or path

    # Training loop
    num_epochs: int = 200
    max_train_steps: Optional[int] = None  # If set, stop after N steps (overrides epochs)
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 1

    # Checkpointing
    save_model_epochs: int = 0  # Save every N epochs (0 = use checkpointing_steps only)
    checkpointing_steps: Optional[int] = None  # Save every N steps
    checkpoints_total_limit: int = 1  # Rotate old checkpoints when exceeding this

    # Logging and validation
    log_every: int = 100
    validation_steps: Optional[int] = None
    validation_epochs: Optional[int] = None
    log_with: str = "tensorboard"  # "tensorboard" | "wandb" | "swanlab"

    # Hardware
    mixed_precision: str = "no"  # "no" | "fp16" | "bf16"
    dataloader_num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
