# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Image translation model architectures."""

from src.models.discriminators import PatchGANDiscriminator
from src.models.generators import ResNetGenerator, UNetGenerator
from src.models.unet import (
    BBDMUNet,
    BDBMUNet,
    BiBBDMUNet,
    CDTSDEUNet,
    DBIMUNet,
    DDBMUNet,
    DDIBUNet,
    I2SBDiffusersUNet,
    I2SBUNet,
    LBMUNet,
    create_model,
)
from src.models.dit import SiTBackbone, SIT_CONFIGS
from src.models.stegogan import (
    ResnetMaskV1Generator,
    ResnetMaskV3Generator,
    NetMatchability,
)
from src.models.cut import (
    CUTGenerator,
    create_generator,
    create_discriminator,
    create_patch_sample_mlp,
)
from src.models.unsb import (
    UNSBGenerator,
    UNSBDiscriminator,
    UNSBEnergyNet,
)
from src.models.local_diffusion import (
    LocalDiffusionUNet,
    ConditionEncoder,
)
from src.models.pix2pixhd import Pix2PixHDGenerator, Pix2PixHDGlobalGenerator
from src.models.stargan import StarGANGenerator, StarGANDiscriminator, StarGANResidualBlock
from src.models.fcdm import (
    FCDM,
    FCDM_S,
    FCDM_B,
    FCDM_L,
    FCDM_XL,
    FCDMImageCond,
    FCDM_MODELS,
)

__all__ = [
    "UNetGenerator",
    "ResNetGenerator",
    "PatchGANDiscriminator",
    "I2SBUNet",
    "BBDMUNet",
    "BDBMUNet",
    "BiBBDMUNet",
    "CDTSDEUNet",
    "DBIMUNet",
    "DDBMUNet",
    "DDIBUNet",
    "I2SBDiffusersUNet",
    "LBMUNet",
    "SiTBackbone",
    "SIT_CONFIGS",
    "ResnetMaskV1Generator",
    "ResnetMaskV3Generator",
    "NetMatchability",
    "CUTGenerator",
    "create_generator",
    "create_discriminator",
    "create_patch_sample_mlp",
    "UNSBGenerator",
    "UNSBDiscriminator",
    "UNSBEnergyNet",
    "LocalDiffusionUNet",
    "ConditionEncoder",
    "Pix2PixHDGenerator",
    "Pix2PixHDGlobalGenerator",
    "StarGANGenerator",
    "StarGANDiscriminator",
    "StarGANResidualBlock",
    "FCDM",
    "FCDM_S",
    "FCDM_B",
    "FCDM_L",
    "FCDM_XL",
    "FCDMImageCond",
    "FCDM_MODELS",
]
