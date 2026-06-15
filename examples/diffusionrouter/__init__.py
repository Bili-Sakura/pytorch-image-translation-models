# Copyright (c) 2026 EarthBridge Team.
# Credits: DiffusionRouter (kvmduc) - https://github.com/kvmduc/DiffusionRouter

"""DiffusionRouter inference example.

Training and distillation use the upstream DiffusionRouter scripts; see
``examples/community/diffusionrouter/train.py``.
"""

from src.models.diffusionrouter import (
    DIFFUSIONROUTER_CLASS_NAMES,
    DIFFUSIONROUTER_DEFAULT_CHAIN,
    DiffusionRouterConfig,
)
from src.pipelines.diffusionrouter import (
    DiffusionRouterPipeline,
    DiffusionRouterPipelineOutput,
    load_diffusionrouter_pipeline,
)

__all__ = [
    "DIFFUSIONROUTER_CLASS_NAMES",
    "DIFFUSIONROUTER_DEFAULT_CHAIN",
    "DiffusionRouterConfig",
    "DiffusionRouterPipeline",
    "DiffusionRouterPipelineOutput",
    "load_diffusionrouter_pipeline",
]
