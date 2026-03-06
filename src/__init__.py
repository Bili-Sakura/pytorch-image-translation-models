"""PyTorch image-to-image translation models.

A library for multi-modal image translation with diffusion bridges,
GANs, and transformer backbones.
"""

__version__ = "0.1.1"

from src.data import PairedImageDataset, UnpairedImageDataset, default_transforms, get_transforms
from src.inference import ImageTranslator
from src.losses import GANLoss, PerceptualLoss
from src.metrics import compute_fid, compute_lpips, compute_psnr, compute_ssim
from src.models import I2SBUNet, PatchGANDiscriminator, ResNetGenerator, SIT_CONFIGS, SiTBackbone, UNetGenerator
from src.pipelines import (
    BDBMPipeline,
    BDBMPipelineOutput,
    BiBBDMPipeline,
    BiBBDMPipelineOutput,
    CDTSDEPipeline,
    CDTSDEPipelineOutput,
    DBIMPipeline,
    DBIMPipelineOutput,
    DDBMPipeline,
    DDBMPipelineOutput,
    DDIBPipeline,
    DDIBPipelineOutput,
    I2SBPipeline,
    I2SBPipelineOutput,
    LBMPipeline,
    LBMPipelineOutput,
)
from src.schedulers import (
    BDBMScheduler,
    BDBMSchedulerOutput,
    BiBBDMScheduler,
    BiBBDMSchedulerOutput,
    CDTSDEScheduler,
    CDTSDESchedulerOutput,
    DBIMScheduler,
    DBIMSchedulerOutput,
    DDBMScheduler,
    DDBMSchedulerOutput,
    DDIBScheduler,
    DDIBSchedulerOutput,
    I2SBScheduler,
    I2SBSchedulerOutput,
    LBMScheduler,
    LBMSchedulerOutput,
)
from src.training import Pix2PixTrainer

__all__ = [
    # Models
    "UNetGenerator",
    "ResNetGenerator",
    "PatchGANDiscriminator",
    "I2SBUNet",
    "SiTBackbone",
    "SIT_CONFIGS",
    # Schedulers
    "BDBMScheduler",
    "BDBMSchedulerOutput",
    "BiBBDMScheduler",
    "BiBBDMSchedulerOutput",
    "CDTSDEScheduler",
    "CDTSDESchedulerOutput",
    "DBIMScheduler",
    "DBIMSchedulerOutput",
    "DDBMScheduler",
    "DDBMSchedulerOutput",
    "DDIBScheduler",
    "DDIBSchedulerOutput",
    "I2SBScheduler",
    "I2SBSchedulerOutput",
    "LBMScheduler",
    "LBMSchedulerOutput",
    # Pipelines
    "BDBMPipeline",
    "BDBMPipelineOutput",
    "BiBBDMPipeline",
    "BiBBDMPipelineOutput",
    "CDTSDEPipeline",
    "CDTSDEPipelineOutput",
    "DBIMPipeline",
    "DBIMPipelineOutput",
    "DDBMPipeline",
    "DDBMPipelineOutput",
    "DDIBPipeline",
    "DDIBPipelineOutput",
    "I2SBPipeline",
    "I2SBPipelineOutput",
    "LBMPipeline",
    "LBMPipelineOutput",
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
