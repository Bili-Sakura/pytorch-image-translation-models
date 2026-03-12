# Copyright (c) 2026 EarthBridge Team.
# Credits: Bosch Research LDDBM (Multimodal-Distribution-Translation-MDT).

"""MDT / LDDBM community pipeline for general modality translation.

Wraps https://github.com/boschresearch/Multimodal-Distribution-Translation-MDT.
Requires the MDT repo to be installed: pip install -e /path/to/Multimodal-Distribution-Translation-MDT

* ``load_mdt_community_pipeline`` – Load MDT pipeline from checkpoint directory.
* ``MDTPipeline`` / ``MDTPipelineOutput`` – Inference pipeline and output types.
"""

from examples.community.mdt.pipeline import (
    MDTPipeline,
    MDTPipelineOutput,
    load_mdt_community_pipeline,
)

__all__ = [
    "MDTPipeline",
    "MDTPipelineOutput",
    "load_mdt_community_pipeline",
]
