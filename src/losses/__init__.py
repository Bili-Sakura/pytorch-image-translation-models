"""Loss functions for image translation tasks."""

from src.losses.adversarial import GANLoss
from src.losses.perceptual import PerceptualLoss

__all__ = [
    "GANLoss",
    "PerceptualLoss",
]
