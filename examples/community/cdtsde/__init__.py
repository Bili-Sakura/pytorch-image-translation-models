# Copyright (c) 2026 EarthBridge Team.
# Credits: CDTSDE/PSCDE (solar defect identification, projects/CDTSDE).
"""CDTSDE community pipeline for ControlLDM (solar defect PSCDE)."""

from examples.community.cdtsde.pipeline import (
    CDTSDECommunityPipeline,
    CDTSDECommunityPipelineOutput,
    load_cdtsde_community_pipeline,
)

__all__ = [
    "CDTSDECommunityPipeline",
    "CDTSDECommunityPipelineOutput",
    "load_cdtsde_community_pipeline",
]
