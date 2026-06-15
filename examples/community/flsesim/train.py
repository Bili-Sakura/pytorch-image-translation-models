# Copyright (c) 2026 EarthBridge Team.
# Credits: F-LSeSim from Zheng et al. CVPR 2021 —
# https://github.com/lyndonzheng/F-LSeSim

"""F-LSeSim training entry point."""

from examples.flsesim.config import FLSeSimConfig
from examples.flsesim.train_flsesim import FLSeSimTrainer

__all__ = ["FLSeSimConfig", "FLSeSimTrainer"]
