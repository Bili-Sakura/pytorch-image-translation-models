# Copyright (c) 2026 EarthBridge Team.
# Credits: Berman et al., NeurIPS 2025 - Bosch Research MDT.

"""Enum names for MDT model configuration."""

from enum import Enum


class Datasets(Enum):
    ShapeNet = "ShapeNet"
    SR = "SuperResolution"


class TrainingStrategy(Enum):
    WholeSystemTraining = "WholeSystemTraining"
    IterativeTraining = "IterativeTraining"


class ReconstructionLoss(Enum):
    Reconstruction = "Reconstruction"
    Predictive = "Predictive"


class DistanceMetric(Enum):
    MSE = "MSE"
    LPIPS = "LPIPS"


class BridgeModelsTyps(Enum):
    BridgeTransformer = "BridgeTransformer"


class Encoders(Enum):
    Conv3DEncoder = "Conv3DEncoder"
    MV2DEncoder = "MV2DEncoder"
    Identity = "Identity"
    KlVaePreTrainedEncoder16 = "KlVaePreTrainedEncoder16"
    KlVaePreTrainedEncoder128 = "KlVaePreTrainedEncoder128"


class Decoders(Enum):
    NoDecoder = "NoDecoder"
    Conv3DDecoder = "Conv3DDecoder"
    KlVaePreTrainedDecoder128 = "KlVaePreTrainedDecoder128"
    Identity = "Identity"
