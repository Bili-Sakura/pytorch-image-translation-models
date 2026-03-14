# Copyright (c) 2026 EarthBridge Team.
"""LBM training for flow-matching bridge."""

from .config import LBMConfig
from .train_lbm import LBMTrainer

__all__ = ["LBMConfig", "LBMTrainer"]
