# Copyright (c) 2026 EarthBridge Team.
# Credits: alexzhou907/DDBM (Zhou et al., ICLR 2024).

"""DDBM community pipeline for OpenAI-style checkpoints (BiliSakura/DDBM-ckpt).

The standard DDBM wrapper uses diffusers UNet2DModel, which does not match
the architecture of alexzhou907/DDBM (improved_diffusion format). This community
package provides:

* ``OpenAIDDBMUNet`` – OpenAI-style UNet compatible with DDBM checkpoints.
* ``load_ddbm_community_pipeline`` – Load DDBM pipeline from unet/ format.
* ``convert_pt_to_unet`` – Convert raw .pt to unet/config.json + safetensors.

See ``README.md`` for usage.
"""

from examples.community.ddbm.model import OpenAIDDBMUNet
from examples.community.ddbm.pipeline import load_ddbm_community_pipeline

__all__ = [
    "OpenAIDDBMUNet",
    "load_ddbm_community_pipeline",
]
