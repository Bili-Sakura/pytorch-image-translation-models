# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from E3Diff (Qin et al., IEEE GRSL 2024).
# Original: https://github.com/DeepSARRS/E3Diff

"""E3Diff – Efficient End-to-End Diffusion Model for SAR-to-Optical Image Translation.

This is a **community pipeline** bundling all network definitions, diffusion
logic, losses, and a training helper.

Submodules
----------
``model``
    Network architectures (``E3DiffUNet``, ``CPEN``, ``GaussianDiffusion``,
    discriminator, losses).
``pipeline``
    Configuration (``E3DiffConfig``) and trainer (``E3DiffTrainer``).
"""

from .model import (
    CPEN,
    E3DiffUNet,
    FocalFrequencyLoss,
    GaussianDiffusion,
    _GANLoss,
    _init_weights,
)
from .pipeline import E3DiffConfig, E3DiffTrainer

__all__ = [
    "CPEN",
    "E3DiffUNet",
    "FocalFrequencyLoss",
    "GaussianDiffusion",
    "E3DiffConfig",
    "E3DiffTrainer",
    "_GANLoss",
    "_init_weights",
]
