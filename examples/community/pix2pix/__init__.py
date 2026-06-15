# Credits: pix2pix (Isola et al., CVPR 2017) — https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""pix2pix community pipeline."""

from src.pipelines.pix2pix import (
    ImageTranslator,
    Pix2PixPipeline,
    Pix2PixPipelineOutput,
)

from .model import Pix2PixGenerator, create_generator, create_discriminator
from .pipeline import load_pix2pix_community_pipeline, load_pix2pix_pipeline
from .train import Pix2PixConfig, Pix2PixTrainer, TrainingConfig

__all__ = [
    "Pix2PixGenerator",
    "create_generator",
    "create_discriminator",
    "Pix2PixPipeline",
    "Pix2PixPipelineOutput",
    "ImageTranslator",
    "Pix2PixConfig",
    "Pix2PixTrainer",
    "TrainingConfig",
    "load_pix2pix_pipeline",
    "load_pix2pix_community_pipeline",
]
