"""BBDM (Brownian Bridge Diffusion Model) training example.

Uses src.pipelines.bbdm, src.schedulers.bbdm, src.models.unet.BBDMUNet.
"""

from examples.bbdm.config import BBDMConfig
from examples.bbdm.train_bbdm import BBDMTrainer

__all__ = ["BBDMConfig", "BBDMTrainer"]
