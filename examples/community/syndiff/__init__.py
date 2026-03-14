# Copyright (c) 2026 EarthBridge Team.
# Credits: SynDiff (icon-lab, IEEE TMI 2023) - https://github.com/icon-lab/SynDiff

"""SynDiff community pipeline for unsupervised medical image translation."""

from examples.community.syndiff.pipeline import (
    SynDiffPipeline,
    SynDiffPipelineOutput,
    load_syndiff_community_pipeline,
)

__all__ = [
    "SynDiffPipeline",
    "SynDiffPipelineOutput",
    "load_syndiff_community_pipeline",
]
