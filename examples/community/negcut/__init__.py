# Copyright (c) 2026 EarthBridge Team.
# Credits: NEGCUT from Wang et al. ICCV 2021 — https://github.com/WeilunWang/NEGCUT

"""NEGCUT community pipeline — re-exports core library components."""

from src.models.negcut import (
    FeatureNormalize,
    LearnedPatchNCELoss,
    NegativeGenerator,
    NegativePlaceholder,
    create_negative_generator,
)
from src.pipelines.cut import NEGCUTPipeline, NEGCUTPipelineOutput

from .pipeline import load_negcut_pipeline
from .train import NEGCUTConfig, NEGCUTTrainer

__all__ = [
    "FeatureNormalize",
    "LearnedPatchNCELoss",
    "NegativeGenerator",
    "NegativePlaceholder",
    "create_negative_generator",
    "NEGCUTPipeline",
    "NEGCUTPipelineOutput",
    "NEGCUTConfig",
    "NEGCUTTrainer",
    "load_negcut_pipeline",
]
