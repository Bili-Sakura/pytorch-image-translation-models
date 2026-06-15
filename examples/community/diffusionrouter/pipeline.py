# Copyright (c) 2026 EarthBridge Team.
# Credits: DiffusionRouter (kvmduc) - https://github.com/kvmduc/DiffusionRouter

"""DiffusionRouter community loader — wraps the core pipeline in ``src``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.models.diffusionrouter import DiffusionRouterConfig
from src.pipelines.diffusionrouter import (
    DiffusionRouterPipeline,
    DiffusionRouterPipelineOutput,
    load_diffusionrouter_pipeline,
)


def load_diffusionrouter_community_pipeline(
    checkpoint_path: str | Path,
    *,
    diffusionrouter_src_path: Optional[str] = None,
    config: Optional[DiffusionRouterConfig] = None,
    device: Optional[str] = None,
) -> DiffusionRouterPipeline:
    """Load DiffusionRouter pipeline from a ``.pt`` checkpoint.

    Alias for :func:`src.pipelines.diffusionrouter.load_diffusionrouter_pipeline`.
    """
    return load_diffusionrouter_pipeline(
        checkpoint_path,
        diffusionrouter_src_path=diffusionrouter_src_path,
        config=config,
        device=device,
    )


__all__ = [
    "DiffusionRouterPipeline",
    "DiffusionRouterPipelineOutput",
    "load_diffusionrouter_community_pipeline",
    "load_diffusionrouter_pipeline",
]
