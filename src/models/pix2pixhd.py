"""pix2pixHD generator architectures.

Implements the core generator used in NVIDIA pix2pixHD:

- ``Pix2PixHDGlobalGenerator``: global coarse-to-fine ResNet generator.
- ``Pix2PixHDGenerator``: alias wrapper around the global generator for a
  consistent project-level naming convention.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn


class _ResnetBlock(nn.Module):
    """A residual block used by pix2pixHD generators."""

    def __init__(
        self,
        dim: int,
        padding_type: str = "reflect",
        norm_layer: type[nn.Module] | functools.partial = nn.InstanceNorm2d,
        activation: nn.Module | None = None,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.ReLU(True)
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        conv_block: list[nn.Module] = []
        p = 0
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise ValueError(f"Unsupported padding type: {padding_type}")

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            activation,
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class Pix2PixHDGlobalGenerator(nn.Module):
    """Global generator from pix2pixHD.

    This module is compatible with the common ``netG`` checkpoint produced by
    the original NVIDIA pix2pixHD training scripts.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        n_downsampling: int = 4,
        n_blocks: int = 9,
        norm_layer: type[nn.Module] | functools.partial = nn.InstanceNorm2d,
        padding_type: str = "reflect",
    ) -> None:
        super().__init__()
        if n_blocks < 0:
            raise ValueError("n_blocks must be >= 0")
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)
        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            activation,
        ]

        # Downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                activation,
            ]

        # Resnet blocks
        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [_ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        # Upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                activation,
            ]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Pix2PixHDGenerator(Pix2PixHDGlobalGenerator):
    """Project-level pix2pixHD generator alias.

    The initial baseline integration exposes the global generator as the
    default pix2pixHD implementation.
    """

