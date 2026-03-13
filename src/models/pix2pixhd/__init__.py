# Credits: pix2pixHD from Wang et al. "High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs" CVPR 2018.
#
"""pix2pixHD generator architectures.

Implements the core generator used in NVIDIA pix2pixHD:

- ``Pix2PixHDGlobalGenerator``: global coarse-to-fine ResNet generator.
- ``Pix2PixHDGenerator``: alias wrapper around the global generator for a
  consistent project-level naming convention.
"""

from src.models.pix2pixhd.generator import Pix2PixHDGenerator, Pix2PixHDGlobalGenerator

__all__ = [
    "Pix2PixHDGenerator",
    "Pix2PixHDGlobalGenerator",
]

