# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from Parallel-GAN (Wang et al., TGRS 2022).

"""Parallel-GAN – SAR-to-Optical image translation with hierarchical latent features.

This community pipeline is organised into three modules:

* ``model.py`` – Network architectures (``ParaGAN``, ``Resrecon``, discriminator, losses).
* ``pipeline.py`` – Inference pipeline (``ParallelGANPipeline``, inherits ``DiffusionPipeline``).
* ``train.py``  – Training configuration (``ParallelGANConfig``) and harness (``ParallelGANTrainer``).

See ``readme.md`` in this directory for usage examples and citation info.
"""

from examples.community.parallel_gan.model import (
    ParaGAN,
    Resrecon,
    VGGLoss,
    _GANLoss,
)
from examples.community.parallel_gan.pipeline import (
    ParallelGANPipeline,
    ParallelGANPipelineOutput,
)
from examples.community.parallel_gan.train import (
    ParallelGANConfig,
    ParallelGANTrainer,
)

__all__ = [
    "ParaGAN",
    "Resrecon",
    "VGGLoss",
    "_GANLoss",
    "ParallelGANPipeline",
    "ParallelGANPipelineOutput",
    "ParallelGANConfig",
    "ParallelGANTrainer",
]
