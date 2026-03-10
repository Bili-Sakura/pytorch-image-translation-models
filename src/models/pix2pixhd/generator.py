"""pix2pixHD generator architectures."""

from __future__ import annotations

import functools

import torch
import torch.nn as nn

from src.models.pix2pixhd.blocks import _ResnetBlock


class Pix2PixHDGlobalGenerator(nn.Module):
    """Global generator from pix2pixHD."""

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

        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                activation,
            ]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [_ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

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
    """Project-level pix2pixHD generator alias."""

