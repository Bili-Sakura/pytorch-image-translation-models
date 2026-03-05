"""PyTorch image-to-image translation models."""

__version__ = "0.1.0"

from src.data import PairedImageDataset, UnpairedImageDataset, default_transforms, get_transforms
from src.inference import ImageTranslator
from src.losses import GANLoss, PerceptualLoss
from src.metrics import compute_fid, compute_lpips, compute_psnr, compute_ssim
from src.models import PatchGANDiscriminator, ResNetGenerator, UNetGenerator
from src.training import Pix2PixTrainer

__all__ = [
    # Models
    "UNetGenerator",
    "ResNetGenerator",
    "PatchGANDiscriminator",
    # Data
    "PairedImageDataset",
    "UnpairedImageDataset",
    "get_transforms",
    "default_transforms",
    # Losses
    "GANLoss",
    "PerceptualLoss",
    # Training
    "Pix2PixTrainer",
    # Inference
    "ImageTranslator",
    # Metrics
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    "compute_fid",
]
