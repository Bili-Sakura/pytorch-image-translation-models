"""Loss functions for image translation tasks."""

from src.losses.adversarial import GANLoss
from src.losses.diffusion import DiffusionLoss, cosine_interpolated_logsnr, get_diffusion_loss
from src.losses.perceptual import PerceptualLoss

__all__ = [
    "GANLoss",
    "PerceptualLoss",
    "DiffusionLoss",
    "cosine_interpolated_logsnr",
    "get_diffusion_loss",
]
