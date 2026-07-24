# Credits: SPADE from Park et al. "Semantic Image Synthesis with Spatially-Adaptive Normalization" CVPR 2019.
#
"""SPADE generator architecture."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from src.models.spade.blocks import SPADEResnetBlock


def _num_upsampling_layers(num_upsampling_layers: str) -> int:
    if num_upsampling_layers == "normal":
        return 5
    if num_upsampling_layers == "more":
        return 6
    if num_upsampling_layers == "most":
        return 7
    raise ValueError(f"Unsupported num_upsampling_layers: {num_upsampling_layers}")


def compute_latent_spatial_size(
    crop_size: int,
    *,
    num_upsampling_layers: str = "normal",
    aspect_ratio: float = 1.0,
) -> tuple[int, int]:
    """Return latent (width, height) for a SPADE generator."""
    num_up = _num_upsampling_layers(num_upsampling_layers)
    sw = crop_size // (2**num_up)
    sh = max(1, round(sw / aspect_ratio))
    return sw, sh


class SPADEGenerator(ModelMixin, ConfigMixin):
    """SPADE generator for semantic image synthesis.

    Takes a semantic segmentation map and produces a photorealistic image.
    Optionally supports VAE-style random style sampling via ``use_vae``.
    """

    @register_to_config
    def __init__(
        self,
        label_nc: int = 35,
        output_nc: int = 3,
        ngf: int = 64,
        crop_size: int = 256,
        aspect_ratio: float = 1.0,
        num_upsampling_layers: Literal["normal", "more", "most"] = "normal",
        use_vae: bool = False,
        z_dim: int = 256,
        use_spectral_norm: bool = True,
        param_free_norm_type: str = "instance",
        spade_hidden: int = 128,
        spade_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.label_nc = label_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.crop_size = crop_size
        self.aspect_ratio = aspect_ratio
        self.num_upsampling_layers = num_upsampling_layers
        self.use_vae = use_vae
        self.z_dim = z_dim

        self.sw, self.sh = compute_latent_spatial_size(
            crop_size,
            num_upsampling_layers=num_upsampling_layers,
            aspect_ratio=aspect_ratio,
        )

        block_kwargs = {
            "label_nc": label_nc,
            "use_spectral_norm": use_spectral_norm,
            "param_free_norm_type": param_free_norm_type,
            "spade_hidden": spade_hidden,
            "spade_kernel_size": spade_kernel_size,
        }

        if use_vae:
            self.fc = nn.Linear(z_dim, 16 * ngf * self.sw * self.sh)
        else:
            conv = nn.Conv2d(label_nc, 16 * ngf, kernel_size=3, padding=1)
            self.fc = spectral_norm(conv) if use_spectral_norm else conv

        self.head_0 = SPADEResnetBlock(16 * ngf, 16 * ngf, **block_kwargs)
        self.G_middle_0 = SPADEResnetBlock(16 * ngf, 16 * ngf, **block_kwargs)
        self.G_middle_1 = SPADEResnetBlock(16 * ngf, 16 * ngf, **block_kwargs)
        self.up_0 = SPADEResnetBlock(16 * ngf, 8 * ngf, **block_kwargs)
        self.up_1 = SPADEResnetBlock(8 * ngf, 4 * ngf, **block_kwargs)
        self.up_2 = SPADEResnetBlock(4 * ngf, 2 * ngf, **block_kwargs)
        self.up_3 = SPADEResnetBlock(2 * ngf, ngf, **block_kwargs)

        final_nc = ngf
        if num_upsampling_layers == "most":
            self.up_4 = SPADEResnetBlock(ngf, ngf // 2, **block_kwargs)
            final_nc = ngf // 2

        conv_img = nn.Conv2d(final_nc, output_nc, kernel_size=3, padding=1)
        self.conv_img = spectral_norm(conv_img) if use_spectral_norm else conv_img
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(
        self,
        segmap: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate an image from a semantic segmentation map."""
        if self.use_vae:
            if z is None:
                z = torch.randn(segmap.size(0), self.z_dim, device=segmap.device, dtype=segmap.dtype)
            x = self.fc(z)
            x = x.view(-1, 16 * self.ngf, self.sh, self.sw)
        else:
            x = F.interpolate(segmap, size=(self.sh, self.sw), mode="nearest")
            x = self.fc(x)

        x = self.head_0(x, segmap)

        x = self.up(x)
        x = self.G_middle_0(x, segmap)

        if self.num_upsampling_layers in ("more", "most"):
            x = self.up(x)

        x = self.G_middle_1(x, segmap)

        x = self.up(x)
        x = self.up_0(x, segmap)
        x = self.up(x)
        x = self.up_1(x, segmap)
        x = self.up(x)
        x = self.up_2(x, segmap)
        x = self.up(x)
        x = self.up_3(x, segmap)

        if self.num_upsampling_layers == "most":
            x = self.up(x)
            x = self.up_4(x, segmap)

        x = self.conv_img(F.leaky_relu(x, 0.2))
        return torch.tanh(x)
