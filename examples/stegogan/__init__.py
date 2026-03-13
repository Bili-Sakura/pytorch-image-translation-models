# Credits: Adapted from StegoGAN (Wu et al., CVPR 2024). Original: https://github.com/sian-wusidi/StegoGAN
"""StegoGAN example: non-bijective unpaired image translation training."""

from examples.stegogan.train_stegogan import StegoGANConfig, StegoGANTrainer

__all__ = [
    "StegoGANTrainer",
    "StegoGANConfig",
]
