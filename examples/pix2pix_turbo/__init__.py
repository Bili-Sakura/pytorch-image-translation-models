# Copyright (c) 2026 EarthBridge Team.
# Credits: img2img-turbo from Parmar et al. 2024 —
# https://github.com/GaParmar/img2img-turbo

"""pix2pix-turbo training entry point."""

from examples.pix2pix_turbo.config import Pix2PixTurboConfig
from examples.pix2pix_turbo.train_pix2pix_turbo import Pix2PixTurboTrainer, main

__all__ = ["Pix2PixTurboConfig", "Pix2PixTurboTrainer", "main"]
