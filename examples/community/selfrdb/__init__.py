# Copyright (c) 2026 EarthBridge Team.
# Credits: SelfRDB (Arslan et al., Medical Image Analysis 2024).

"""SelfRDB community pipeline for medical image translation.

Self-Consistent Recursive Diffusion Bridge. Uses load_selfrdb_community_pipeline
to load from original .ckpt files.
"""

from examples.community.selfrdb.pipeline import (
    SelfRDBPipeline,
    SelfRDBPipelineOutput,
    load_selfrdb_community_pipeline,
)

__all__ = [
    "SelfRDBPipeline",
    "SelfRDBPipelineOutput",
    "load_selfrdb_community_pipeline",
]
