# Copyright (c) 2026 EarthBridge Team.
# Credits: Decent from Xie et al. NeurIPS 2022 — https://github.com/Mid-Push/Decent

"""Decent training entry point."""

from examples.decent.config import DecentConfig
from examples.decent.train_decent import DecentTrainer

__all__ = ["DecentConfig", "DecentTrainer"]
