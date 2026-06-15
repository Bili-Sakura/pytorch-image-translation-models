# Copyright (c) 2026 EarthBridge Team.
# Credits: F-LSeSim from Zheng et al. CVPR 2021 —
# https://github.com/lyndonzheng/F-LSeSim

"""F-LSeSim community pipeline — re-exports core library components."""

from src.models.flsesim import (
    ImageNetNormalization,
    PatchSim,
    SpatialCorrelativeLoss,
    VGG16FeatureExtractor,
    compute_spatial_correlative_loss,
)
from src.pipelines.cut import FLSeSimPipeline, FLSeSimPipelineOutput

from .pipeline import load_flsesim_pipeline
from .train import FLSeSimConfig, FLSeSimTrainer

__all__ = [
    "ImageNetNormalization",
    "PatchSim",
    "SpatialCorrelativeLoss",
    "VGG16FeatureExtractor",
    "compute_spatial_correlative_loss",
    "FLSeSimPipeline",
    "FLSeSimPipelineOutput",
    "FLSeSimConfig",
    "FLSeSimTrainer",
    "load_flsesim_pipeline",
]
