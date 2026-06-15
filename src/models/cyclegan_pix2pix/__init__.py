# Credits: CycleGAN (Zhu et al., ICCV 2017) and pix2pix (Isola et al., CVPR 2017).
# Adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""CycleGAN and pix2pix model components."""

from src.models.cyclegan_pix2pix.networks import (
    CycleGANGenerator,
    GANLoss,
    Pix2PixGenerator,
    PixelDiscriminator,
    create_discriminator,
    create_generator,
    get_norm_layer,
    init_weights,
    load_upstream_generator_state,
    patch_instance_norm_state_dict,
)

__all__ = [
    "CycleGANGenerator",
    "Pix2PixGenerator",
    "PixelDiscriminator",
    "GANLoss",
    "get_norm_layer",
    "init_weights",
    "create_generator",
    "create_discriminator",
    "load_upstream_generator_state",
    "patch_instance_norm_state_dict",
]
