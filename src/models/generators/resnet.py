"""ResNet generator architecture."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and a skip connection."""

    def __init__(self, channels: int, norm_layer: type[nn.Module] = nn.InstanceNorm2d) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            norm_layer(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            norm_layer(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """ResNet-based generator for image-to-image translation."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_filters: int = 64,
        n_residual_blocks: int = 9,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
    ) -> None:
        super().__init__()

        encoder: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_filters, kernel_size=7, bias=False),
            norm_layer(base_filters),
            nn.ReLU(inplace=True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [
                nn.Conv2d(
                    base_filters * mult,
                    base_filters * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                norm_layer(base_filters * mult * 2),
                nn.ReLU(inplace=True),
            ]

        mult = 2**n_downsampling
        residual = [
            ResidualBlock(base_filters * mult, norm_layer=norm_layer)
            for _ in range(n_residual_blocks)
        ]

        decoder: list[nn.Module] = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder += [
                nn.ConvTranspose2d(
                    base_filters * mult,
                    base_filters * mult // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                norm_layer(base_filters * mult // 2),
                nn.ReLU(inplace=True),
            ]

        decoder += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_filters, out_channels, kernel_size=7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*encoder, *residual, *decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

