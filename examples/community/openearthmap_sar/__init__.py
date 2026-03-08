# Copyright (c) 2026 EarthBridge Team.
# Credits: OpenEarthMap-SAR CUT models. Architecture from Park et al. "Contrastive Learning for Unpaired Image-to-Image Translation" ECCV 2020.

"""OpenEarthMap-SAR – CUT models for SAR ↔ optical image translation.

This community pipeline enables loading and inference of OpenEarthMap-SAR checkpoints
using pytorch-image-translation-models. The checkpoints use the original CUT architecture
with anti-aliased down/upsampling, which differs from the default CUTGenerator in
src.models.cut.

* ``model.py``    – OpenEarthMapSARGenerator (CUT ResNet with anti-aliased down/upsampling).
* ``pipeline.py`` – load_openearthmap_sar_pipeline (loads checkpoints, returns CUTPipeline).
* ``train.py``    – Stub; models are pre-trained (see main CUT training).
"""

from .model import OpenEarthMapSARGenerator
from .pipeline import OPENEARTHMAP_SAR_MODELS, load_openearthmap_sar_pipeline

__all__ = [
    "OpenEarthMapSARGenerator",
    "OPENEARTHMAP_SAR_MODELS",
    "load_openearthmap_sar_pipeline",
]
