# Credits: StarGAN (Choi et al., CVPR 2018) - https://github.com/yunjey/stargan

"""StarGAN model architectures.

Adapted from the official implementation:
https://github.com/yunjey/stargan
"""

from src.models.stargan.blocks import StarGANResidualBlock
from src.models.stargan.discriminator import StarGANDiscriminator
from src.models.stargan.generator import StarGANGenerator

__all__ = [
    "StarGANResidualBlock",
    "StarGANGenerator",
    "StarGANDiscriminator",
]

