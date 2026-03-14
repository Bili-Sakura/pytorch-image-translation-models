# Credits: SAR2Optical (https://github.com/yuuIind/SAR2Optical), pix2pix (Isola et al.).
"""SAR2Optical community pipeline (pix2pix-style SAR-to-optical translation).

Adapted from:
https://github.com/yuuIind/SAR2Optical
"""

from examples.community.sar2optical.model import (
    SAR2OpticalDiscriminator,
    SAR2OpticalGenerator,
    init_weights,
)
from examples.community.sar2optical.pipeline import (
    SAR2OpticalPipeline,
    SAR2OpticalPipelineOutput,
    load_sar2optical_pipeline,
)
from examples.community.sar2optical.train import (
    SAR2OpticalConfig,
    SAR2OpticalTrainer,
)

__all__ = [
    "SAR2OpticalGenerator",
    "SAR2OpticalDiscriminator",
    "init_weights",
    "SAR2OpticalPipeline",
    "SAR2OpticalPipelineOutput",
    "load_sar2optical_pipeline",
    "SAR2OpticalConfig",
    "SAR2OpticalTrainer",
]
