# Credits: SPADE from Park et al. "Semantic Image Synthesis with Spatially-Adaptive Normalization" CVPR 2019.
#
"""Style encoder for SPADE VAE training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


def _norm_conv(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int,
    stride: int,
    use_spectral_norm: bool,
) -> nn.Module:
    padding = (kernel_size - 1) // 2
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    if use_spectral_norm:
        conv = spectral_norm(conv)
    return conv


class SPADEStyleEncoder(ModelMixin, ConfigMixin):
    """Convolutional encoder that maps images to a Gaussian style code."""

    @register_to_config
    def __init__(
        self,
        input_nc: int = 3,
        ngf: int = 64,
        z_dim: int = 256,
        crop_size: int = 256,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.crop_size = crop_size

        kw = 3
        self.layer1 = _norm_conv(input_nc, ngf, kernel_size=kw, stride=2, use_spectral_norm=use_spectral_norm)
        self.layer2 = _norm_conv(ngf, ngf * 2, kernel_size=kw, stride=2, use_spectral_norm=use_spectral_norm)
        self.layer3 = _norm_conv(ngf * 2, ngf * 4, kernel_size=kw, stride=2, use_spectral_norm=use_spectral_norm)
        self.layer4 = _norm_conv(ngf * 4, ngf * 8, kernel_size=kw, stride=2, use_spectral_norm=use_spectral_norm)
        self.layer5 = _norm_conv(ngf * 8, ngf * 8, kernel_size=kw, stride=2, use_spectral_norm=use_spectral_norm)
        self.layer6: nn.Module | None = None
        if crop_size >= 256:
            self.layer6 = _norm_conv(ngf * 8, ngf * 8, kernel_size=kw, stride=2, use_spectral_norm=use_spectral_norm)

        self.actvn = nn.LeakyReLU(0.2, inplace=False)

        with torch.no_grad():
            dummy = torch.zeros(1, input_nc, 256, 256)
            flat_dim = self._encode_features(dummy).view(1, -1).shape[1]
        self.fc_mu = nn.Linear(flat_dim, z_dim)
        self.fc_var = nn.Linear(flat_dim, z_dim)

    def _encode_features(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[2] != 256 or image.shape[3] != 256:
            image = F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False)

        x = self.layer1(image)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.layer6 is not None:
            x = self.layer6(self.actvn(x))
        return self.actvn(x)

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._encode_features(image)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return mu, logvar, z

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encode(image)
