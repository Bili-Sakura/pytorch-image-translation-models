# Copyright (c) 2026 EarthBridge Team.
# Credits: F-LSeSim from Zheng et al. CVPR 2021 —
# https://github.com/lyndonzheng/F-LSeSim

"""F-LSeSim spatially-correlative loss components."""

from src.models.flsesim.losses import (
    ImageNetNormalization,
    ImagePool,
    PatchSim,
    SpatialCorrelativeLoss,
    VGG16FeatureExtractor,
    compute_spatial_correlative_loss,
)

__all__ = [
    "ImageNetNormalization",
    "ImagePool",
    "PatchSim",
    "SpatialCorrelativeLoss",
    "VGG16FeatureExtractor",
    "compute_spatial_correlative_loss",
]
