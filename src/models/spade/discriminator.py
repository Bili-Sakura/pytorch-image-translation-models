# Credits: SPADE from Park et al. "Semantic Image Synthesis with Spatially-Adaptive Normalization" CVPR 2019.
#
"""Multi-scale PatchGAN discriminator for SPADE."""

from __future__ import annotations

import functools
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


def _get_norm_layer(norm_type: str = "instance") -> type[nn.Module] | functools.partial:
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        return lambda _: nn.Identity()
    raise ValueError(f"Unsupported norm type: {norm_type}")


class SPADENLayerDiscriminator(nn.Module):
    """PatchGAN discriminator that returns intermediate features."""

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 4,
        *,
        norm_type: str = "instance",
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        norm_layer = _get_norm_layer(norm_type)
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))

        def _conv(in_ch: int, out_ch: int, stride: int) -> nn.Module:
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=kw, stride=stride, padding=padw)
            return spectral_norm(conv) if use_spectral_norm else conv

        sequence: list[list[nn.Module]] = [
            [_conv(input_nc, ndf, 2), nn.LeakyReLU(0.2, inplace=False)],
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == n_layers - 1 else 2
            sequence.append([
                _conv(nf_prev, nf, stride),
                norm_layer(nf),
                nn.LeakyReLU(0.2, inplace=False),
            ])

        sequence.append([_conv(nf, 1, 1)])

        for idx, layers in enumerate(sequence):
            self.add_module(f"model{idx}", nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        results = [x]
        for submodel in self.children():
            results.append(submodel(results[-1]))
        return results[1:]


class SPADEMultiscaleDiscriminator(ModelMixin, ConfigMixin):
    """Multi-scale discriminator operating on concatenated label maps and images."""

    @register_to_config
    def __init__(
        self,
        label_nc: int = 35,
        output_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 4,
        num_D: int = 2,
        *,
        norm_type: str = "instance",
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_D = num_D
        input_nc = label_nc + output_nc
        self.discriminators = nn.ModuleList([
            SPADENLayerDiscriminator(
                input_nc,
                ndf=ndf,
                n_layers=n_layers,
                norm_type=norm_type,
                use_spectral_norm=use_spectral_norm,
            )
            for _ in range(num_D)
        ])

    @staticmethod
    def _downsample(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, label: torch.Tensor, image: torch.Tensor) -> list[list[torch.Tensor]]:
        """Return per-scale discriminator outputs (with intermediate features)."""
        x = torch.cat([label, image], dim=1)
        outputs: List[list[torch.Tensor]] = []
        for discriminator in self.discriminators:
            outputs.append(discriminator(x))
            x = self._downsample(x)
        return outputs
