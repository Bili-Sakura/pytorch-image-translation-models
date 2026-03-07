# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from Parallel-GAN (Wang et al., TGRS 2022).
# Original: https://github.com/ZZG-Z/Parallel-GAN

"""Parallel-GAN – SAR-to-Optical image translation with hierarchical latent features.

This is a **community pipeline** bundling all network definitions, losses,
and a training helper.

Submodules
----------
``model``
    Network architectures (``ParaGAN``, ``Resrecon``, discriminator, losses).
``pipeline``
    Configuration (``ParallelGANConfig``) and trainer (``ParallelGANTrainer``).
"""

from .model import (
    ParaGAN,
    Resrecon,
    VGGLoss,
    _GANLoss,
    _init_weights,
)
from .pipeline import ParallelGANConfig, ParallelGANTrainer

__all__ = [
    "ParaGAN",
    "Resrecon",
    "ParallelGANConfig",
    "ParallelGANTrainer",
    "VGGLoss",
    "_GANLoss",
    "_init_weights",
]
