# Credits: CycleGAN (Zhu et al., ICCV 2017) — https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""CycleGAN community pipeline."""

from src.pipelines.cyclegan import CycleGANPipeline, CycleGANPipelineOutput

from .model import CycleGANGenerator, create_generator, create_discriminator
from .pipeline import load_cyclegan_community_pipeline, load_cyclegan_pipeline
from .train import CycleGANConfig, CycleGANTrainer

__all__ = [
    "CycleGANGenerator",
    "create_generator",
    "create_discriminator",
    "CycleGANPipeline",
    "CycleGANPipelineOutput",
    "CycleGANConfig",
    "CycleGANTrainer",
    "load_cyclegan_pipeline",
    "load_cyclegan_community_pipeline",
]
