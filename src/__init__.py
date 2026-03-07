"""PyTorch image-to-image translation models.

A library for multi-modal image translation with diffusion bridges,
GANs, and transformer backbones.
"""

__version__ = "0.2.2"

from src.data import PairedImageDataset, UnpairedImageDataset, default_transforms, get_transforms
from src.inference import ImageTranslator
from src.losses import GANLoss, PerceptualLoss
from src.metrics import compute_fid, compute_lpips, compute_psnr, compute_ssim
from src.models import I2SBUNet, PatchGANDiscriminator, ResNetGenerator, SIT_CONFIGS, SiTBackbone, UNetGenerator
from src.models import ResnetMaskV1Generator, ResnetMaskV3Generator, NetMatchability
from src.models import BDBMUNet, BiBBDMUNet, CDTSDEUNet, DBIMUNet, DDBMUNet, DDIBUNet, I2SBDiffusersUNet, LBMUNet
from src.models import UNSBGenerator, UNSBDiscriminator, UNSBEnergyNet
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
    StegoGANPipeline,
    StegoGANPipelineOutput,
    UNSBPipeline,
    UNSBPipelineOutput,
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
    UNSBScheduler,
    UNSBSchedulerOutput,
)

__all__ = [
    # Models — native
    "UNetGenerator",
    "ResNetGenerator",
    "PatchGANDiscriminator",
    "I2SBUNet",
    "SiTBackbone",
    "SIT_CONFIGS",
    "ResnetMaskV1Generator",
    "ResnetMaskV3Generator",
    "NetMatchability",
    # Models — UNSB
    "UNSBGenerator",
    "UNSBDiscriminator",
    "UNSBEnergyNet",
    # Models — diffusers UNet wrappers
    "BDBMUNet",
    "BiBBDMUNet",
    "CDTSDEUNet",
    "DBIMUNet",
    "DDBMUNet",
    "DDIBUNet",
    "I2SBDiffusersUNet",
    "LBMUNet",
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
    "UNSBScheduler",
    "UNSBSchedulerOutput",
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
    "StegoGANPipeline",
    "StegoGANPipelineOutput",
    "UNSBPipeline",
    "UNSBPipelineOutput",
    # Data
    "PairedImageDataset",
    "UnpairedImageDataset",
    "get_transforms",
    "default_transforms",
    # Losses
    "GANLoss",
    "PerceptualLoss",
    # Inference
    "ImageTranslator",
    # Metrics
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    "compute_fid",
]
