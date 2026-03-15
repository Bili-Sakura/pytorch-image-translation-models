"""Argparse utilities for training scripts with config-style overrides."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .common import BaseTrainingConfig


def _bool_arg(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def add_training_args(parser: argparse.ArgumentParser, skip: tuple[str, ...] = ()) -> None:
    """Add common training arguments to an ArgumentParser.

    Arguments match BaseTrainingConfig and 4th-MAVIC-T script style.
    Use skip=("output_dir", "device") to avoid duplicates when the parser
    already defines model-specific versions.
    """
    if "output_dir" not in skip:
        parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Checkpoint output directory")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Resume from "latest" or path to checkpoint',
    )
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Stop after N steps (overrides num_epochs)",
    )
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--save_model_epochs",
        type=int,
        default=0,
        help="Save every N epochs (0 = use checkpointing_steps)",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help="Max checkpoints to retain (rotation)",
    )
    parser.add_argument("--log_every", type=int, default=100, help="Log metrics every N steps")
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=None,
        help="Run validation every N steps",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help="Run validation every N epochs",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default="tensorboard",
        help="Logger: tensorboard | wandb | swanlab",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="DataLoader num_workers",
    )
    if "seed" not in skip:
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
    if "device" not in skip:
        parser.add_argument("--device", type=str, default="cuda", help="Device to train on")


def parse_overrides_from_args(
    parser: argparse.ArgumentParser,
    extra_args: dict[str, Callable[[str], Any]] | None = None,
) -> dict[str, Any]:
    """Parse args and return a dict suitable for config overrides.

    Drops None values. Extra type converters can be passed for custom args.
    """
    args = parser.parse_args()
    out = {}
    for k, v in vars(args).items():
        if v is None:
            continue
        if extra_args and k in extra_args:
            out[k] = extra_args[k](v) if isinstance(v, str) else v
        else:
            out[k] = v
    return out
