# Credits: CycleGAN (Zhu et al., ICCV 2017) — https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""CycleGAN training entry point."""

from examples.cyclegan.config import CycleGANConfig
from examples.cyclegan.train_cyclegan import CycleGANTrainer

__all__ = ["CycleGANConfig", "CycleGANTrainer"]
