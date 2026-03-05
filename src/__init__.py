"""PyTorch image-to-image translation models.

A library for multi-modal image translation with diffusion bridges,
GANs, and transformer backbones.
"""

__version__ = "0.1.0"

from src.data import PairedImageDataset, UnpairedImageDataset, default_transforms, get_transforms
from src.inference import ImageTranslator
from src.losses import GANLoss, PerceptualLoss
from src.metrics import compute_fid, compute_lpips, compute_psnr, compute_ssim
from src.models import I2SBUNet, PatchGANDiscriminator, ResNetGenerator, UNetGenerator
from src.pipelines import I2SBPipeline, I2SBPipelineOutput
from src.schedulers import I2SBScheduler, I2SBSchedulerOutput
from src.training import Pix2PixTrainer

__all__ = [
    # Models
    "UNetGenerator",
    "ResNetGenerator",
    "PatchGANDiscriminator",
    "I2SBUNet",
    # Schedulers
    "I2SBScheduler",
    "I2SBSchedulerOutput",
    # Pipelines
    "I2SBPipeline",
    "I2SBPipelineOutput",
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
