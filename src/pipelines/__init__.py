"""Inference pipelines for image translation models."""

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

__all__ = [
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
]
