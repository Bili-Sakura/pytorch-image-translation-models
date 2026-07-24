# Credits: SPADE from Park et al. "Semantic Image Synthesis with Spatially-Adaptive Normalization" CVPR 2019.
#
"""SPADE model components for semantic image synthesis."""

from src.models.spade.blocks import SPADEResnetBlock
from src.models.spade.discriminator import SPADEMultiscaleDiscriminator, SPADENLayerDiscriminator
from src.models.spade.encoder import SPADEStyleEncoder
from src.models.spade.generator import SPADEGenerator, compute_latent_spatial_size
from src.models.spade.normalization import SPADE

__all__ = [
    "SPADE",
    "SPADEResnetBlock",
    "SPADEGenerator",
    "SPADEMultiscaleDiscriminator",
    "SPADENLayerDiscriminator",
    "SPADEStyleEncoder",
    "compute_latent_spatial_size",
]
