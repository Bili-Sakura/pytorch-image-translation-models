# Copyright (c) 2026 EarthBridge Team.

"""BBDM community pipeline for OpenAI-style checkpoints.

The standard ``src`` BBDM wrapper is based on diffusers ``UNet2DModel``.
Original xuekt98/BBDM checkpoints use OpenAI-style UNet keys, so this package
provides a compatible loader.
"""

from examples.community.bbdm.model import OpenAIBBDMUNet
from examples.community.bbdm.pipeline import load_bbdm_community_pipeline

__all__ = [
    "OpenAIBBDMUNet",
    "load_bbdm_community_pipeline",
]
