# Copyright (c) 2026 EarthBridge Team.
# Credits: NEGCUT from Wang et al. ICCV 2021 — https://github.com/WeilunWang/NEGCUT

"""NEGCUT model components (learned negatives + PatchNCE on CUT backbone)."""

from src.models.negcut import (
    FeatureNormalize,
    LearnedPatchNCELoss,
    NegativeGenerator,
    NegativePlaceholder,
    create_negative_generator,
)
from src.models.cut import CUTGenerator, create_generator

__all__ = [
    "FeatureNormalize",
    "LearnedPatchNCELoss",
    "NegativeGenerator",
    "NegativePlaceholder",
    "create_negative_generator",
    "CUTGenerator",
    "create_generator",
]
