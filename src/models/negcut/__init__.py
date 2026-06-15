# Copyright (c) 2026 EarthBridge Team.
# Credits: NEGCUT from Wang et al. ICCV 2021 — https://github.com/WeilunWang/NEGCUT

"""NEGCUT hard-negative contrastive learning components."""

from src.models.negcut.losses import LearnedPatchNCELoss
from src.models.negcut.negative_generator import (
    FeatureNormalize,
    NegativeGenerator,
    NegativePlaceholder,
    create_negative_generator,
)

__all__ = [
    "FeatureNormalize",
    "LearnedPatchNCELoss",
    "NegativeGenerator",
    "NegativePlaceholder",
    "create_negative_generator",
]
