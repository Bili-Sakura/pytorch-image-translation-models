"""Discriminator architectures for image-to-image translation."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator (Markovian discriminator).

    Classifies overlapping image patches as real or fake, producing a
    2-D map of predictions rather than a single scalar.

    Parameters
    ----------
    in_channels:
        Number of input image channels (source + target concatenated).
    base_filters:
        Number of filters in the first convolution layer.
    n_layers:
        Number of intermediate convolution layers.
    norm_layer:
        Normalisation layer constructor.
    """

    def __init__(
        self,
        in_channels: int = 6,
        base_filters: int = 64,
        n_layers: int = 3,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                nn.Conv2d(
                    base_filters * nf_mult_prev,
                    base_filters * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                norm_layer(base_filters * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            nn.Conv2d(
                base_filters * nf_mult_prev,
                base_filters * nf_mult,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            ),
            norm_layer(base_filters * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final 1-channel prediction layer
        layers += [
            nn.Conv2d(base_filters * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
