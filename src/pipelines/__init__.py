"""Inference pipelines for image translation models."""

from src.pipelines.bbdm import BBDMPipeline, BBDMPipelineOutput
from src.pipelines.bdbm import BDBMPipeline, BDBMPipelineOutput
from src.pipelines.bibbdm import BiBBDMPipeline, BiBBDMPipelineOutput
from src.pipelines.cdtsde import CDTSDEPipeline, CDTSDEPipelineOutput
from src.pipelines.dbim import DBIMPipeline, DBIMPipelineOutput
from src.pipelines.ddbm import DDBMPipeline, DDBMPipelineOutput
from src.pipelines.ddib import DDIBPipeline, DDIBPipelineOutput
from src.pipelines.i2sb import I2SBPipeline, I2SBPipelineOutput
from src.pipelines.lbm import LBMPipeline, LBMPipelineOutput
from src.pipelines.stegogan import StegoGANPipeline, StegoGANPipelineOutput
from src.pipelines.cut import CUTPipeline, CUTPipelineOutput
from src.pipelines.unsb import UNSBPipeline, UNSBPipelineOutput
from src.pipelines.local_diffusion import LocalDiffusionPipeline, LocalDiffusionPipelineOutput
from src.pipelines.pix2pixhd import (
    Pix2PixHDPipeline,
    Pix2PixHDPipelineOutput,
    load_pix2pixhd_pipeline,
)
from src.pipelines.stargan import StarGANPipeline, StarGANPipelineOutput, load_stargan_pipeline

__all__ = [
    "BBDMPipeline",
    "BBDMPipelineOutput",
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
    "CUTPipeline",
    "CUTPipelineOutput",
    "UNSBPipeline",
    "UNSBPipelineOutput",
    "LocalDiffusionPipeline",
    "LocalDiffusionPipelineOutput",
    "Pix2PixHDPipeline",
    "Pix2PixHDPipelineOutput",
    "load_pix2pixhd_pipeline",
    "StarGANPipeline",
    "StarGANPipelineOutput",
    "load_stargan_pipeline",
]
