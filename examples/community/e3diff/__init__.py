# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from E3Diff (Qin et al., IEEE GRSL 2024).

"""E3Diff – Efficient End-to-End Diffusion Model for SAR-to-Optical Image Translation.

This community pipeline is organised into three modules:

* ``model.py``    – Network architectures (``E3DiffUNet``, ``CPEN``, losses, building blocks).
* ``pipeline.py`` – Diffusion pipeline (``GaussianDiffusion``, beta schedules).
* ``train.py``    – Training configuration (``E3DiffConfig``) and harness (``E3DiffTrainer``).

See ``readme.md`` in this directory for usage examples and citation info.
"""

from examples.community.e3diff.model import (
    CPEN,
    E3DiffUNet,
    FocalFrequencyLoss,
)
from examples.community.e3diff.pipeline import (
    GaussianDiffusion,
)
from examples.community.e3diff.train import (
    E3DiffConfig,
    E3DiffTrainer,
)

__all__ = [
    "CPEN",
    "E3DiffUNet",
    "FocalFrequencyLoss",
    "GaussianDiffusion",
    "E3DiffConfig",
    "E3DiffTrainer",
]
