# Credits: pix2pix (Isola et al., CVPR 2017) — https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""pix2pix training entry point."""

from examples.pix2pix.train_pix2pix import Pix2PixTrainer, TrainingConfig

Pix2PixConfig = TrainingConfig

__all__ = ["Pix2PixConfig", "Pix2PixTrainer", "TrainingConfig"]
