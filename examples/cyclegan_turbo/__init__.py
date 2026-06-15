# Copyright (c) 2026 EarthBridge Team.
# Credits: img2img-turbo from Parmar et al. 2024 —
# https://github.com/GaParmar/img2img-turbo

"""CycleGAN-Turbo training entry point."""

from examples.cyclegan_turbo.config import CycleGANTurboConfig
from examples.cyclegan_turbo.train_cyclegan_turbo import CycleGANTurboTrainer, main

__all__ = ["CycleGANTurboConfig", "CycleGANTurboTrainer", "main"]
