# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
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
from src.pipelines.cut import (
    CUTPipeline,
    CUTPipelineOutput,
    DecentPipeline,
    DecentPipelineOutput,
    FLSeSimPipeline,
    FLSeSimPipelineOutput,
    HnegSRCPipeline,
    HnegSRCPipelineOutput,
    NEGCUTPipeline,
    NEGCUTPipelineOutput,
)
from src.pipelines.unsb import UNSBPipeline, UNSBPipelineOutput
from src.pipelines.local_diffusion import LocalDiffusionPipeline, LocalDiffusionPipelineOutput
from src.pipelines.pix2pixhd import (
    Pix2PixHDPipeline,
    Pix2PixHDPipelineOutput,
    load_pix2pixhd_pipeline,
)
from src.pipelines.stargan import StarGANPipeline, StarGANPipelineOutput, load_stargan_pipeline
from src.pipelines.ecsi import ECSIPipeline, ECSIPipelineOutput, load_ecsi_pipeline
from src.pipelines.fcdm import (
    FCDMPipeline,
    FCDMPipelineOutput,
    FCDMImageCondPipeline,
)
from src.pipelines.lddbm import (
    LDDBMPipeline,
    LDDBMPipelineOutput,
    load_lddbm_pipeline,
)
from src.pipelines.cyclegan import CycleGANPipeline, CycleGANPipelineOutput, load_cyclegan_pipeline
from src.pipelines.cyclegan_turbo import CycleGANTurboPipeline, CycleGANTurboPipelineOutput
from src.pipelines.pix2pix_turbo import Pix2PixTurboPipeline, Pix2PixTurboPipelineOutput
from src.pipelines.pix2pix import (
    ImageTranslator,
    Pix2PixPipeline,
    Pix2PixPipelineOutput,
    load_pix2pix_pipeline,
)
from src.pipelines.diffusionrouter import (
    DiffusionRouterPipeline,
    DiffusionRouterPipelineOutput,
    load_diffusionrouter_pipeline,
)

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
    "DecentPipeline",
    "DecentPipelineOutput",
    "HnegSRCPipeline",
    "HnegSRCPipelineOutput",
    "NEGCUTPipeline",
    "NEGCUTPipelineOutput",
    "FLSeSimPipeline",
    "FLSeSimPipelineOutput",
    "CycleGANTurboPipeline",
    "CycleGANTurboPipelineOutput",
    "Pix2PixTurboPipeline",
    "Pix2PixTurboPipelineOutput",
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
    "ECSIPipeline",
    "ECSIPipelineOutput",
    "load_ecsi_pipeline",
    "FCDMPipeline",
    "FCDMPipelineOutput",
    "FCDMImageCondPipeline",
    "LDDBMPipeline",
    "LDDBMPipelineOutput",
    "load_lddbm_pipeline",
    "ImageTranslator",
    "Pix2PixPipeline",
    "Pix2PixPipelineOutput",
    "load_pix2pix_pipeline",
    "CycleGANPipeline",
    "CycleGANPipelineOutput",
    "load_cyclegan_pipeline",
    "DiffusionRouterPipeline",
    "DiffusionRouterPipelineOutput",
    "load_diffusionrouter_pipeline",
]
