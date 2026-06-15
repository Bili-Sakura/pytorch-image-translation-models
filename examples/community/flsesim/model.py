# Copyright (c) 2026 EarthBridge Team.
# Credits: F-LSeSim from Zheng et al. CVPR 2021 —
# https://github.com/lyndonzheng/F-LSeSim

"""F-LSeSim model components (spatially-correlative loss on ResNet backbone)."""

from src.models.cut import CUTGenerator, create_generator
from src.models.flsesim import (
    ImageNetNormalization,
    PatchSim,
    SpatialCorrelativeLoss,
    VGG16FeatureExtractor,
    compute_spatial_correlative_loss,
)

__all__ = [
    "ImageNetNormalization",
    "PatchSim",
    "SpatialCorrelativeLoss",
    "VGG16FeatureExtractor",
    "compute_spatial_correlative_loss",
    "CUTGenerator",
    "create_generator",
]
