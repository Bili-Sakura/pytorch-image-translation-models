# Copyright (c) 2026 EarthBridge Team.
"""DDIB training for dual-domain diffusion bridge."""

from .config import DDIBConfig
from .train_ddib import DDIBTrainer

__all__ = ["DDIBConfig", "DDIBTrainer"]
